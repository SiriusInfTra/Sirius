#include "utils/xassert.h"
#include "sched/protocol/event.h"
#include "preempt/xqueue/xqueue.h"
#include "preempt/event/dispatcher.h"

using namespace xsched::sched;
using namespace xsched::preempt;

XQueue::XQueue(std::shared_ptr<HalQueue> hal_queue,
               XQueueHandle handle,
               XDevice device,
               XPreemptMode mode,
               int64_t queue_length,
               int64_t sync_interval)
    : kHandle(handle)
    , kDevice(device)
    , kPreemptMode(mode)
    , hal_queue_(hal_queue)
    , wait_queue_(std::make_shared<WaitQueue>(handle))
    , submit_worker_(std::make_shared<Worker>(hal_queue_,
                                              wait_queue_,
                                              mode,
                                              queue_length,
                                              sync_interval))
{
    XASSERT(kPreemptMode > kPreemptModeDisabled &&
            kPreemptMode < kPreemptModeMax,
            "invalid preempt mode: %d", kPreemptMode);

    auto sync_command = wait_queue_->EnqueueSynchronizeCommand();
    sync_command->Synchronize();

    auto e = std::make_unique<XQueueCreateEvent>(kHandle, kDevice);
    g_event_dispatcher.Dispatch(std::move(e));
}

XQueue::~XQueue()
{
    // If the xqueue is terminated, it should not be preempted.
    terminated_.store(true);
    Resume(true);

    auto destroy_command = std::make_shared<XQueueDestroyCommand>();
    wait_queue_->Enqueue(destroy_command);
    destroy_command->Synchronize();

    auto e = std::make_unique<XQueueDestroyEvent>(kHandle);
    g_event_dispatcher.Dispatch(std::move(e));
}

void XQueue::Submit(std::shared_ptr<HalCommand> hal_command)
{
    hal_queue_->OnSubmit(hal_command);
    hal_command->SetXQueue(shared_from_this());
    hal_command->SetIdx(next_hal_command_idx_.fetch_add(1));
    wait_queue_->Enqueue(hal_command);
}

void XQueue::Synchronize()
{
    auto sync_command = wait_queue_->EnqueueSynchronizeCommand();
    sync_command->Synchronize();
}

void XQueue::Synchronize(std::shared_ptr<HalCommand> hal_command)
{
    if (!hal_command->HalSynchronizable()) {
        // If it is not hal-synchronizable, it can only be synced
        // by waiting its state turns to kCommandStateCompleted.
        hal_command->WaitUntil(kCommandStateCompleted);
        return;
    }
    
    // If this hal command is hal-synchronizable,
    // then use submit worker to sync it.
    hal_command->WaitUntil(kCommandStateHalSubmmited);
    submit_worker_->SynchronizeCommand(hal_command);
}

void XQueue::Suspend(bool sync_hal_queue)
{
    // If the xqueue is already terminated, it should not be suspended.
    if (terminated_.load()) return;

    // Will not suspend if it is already suspended.
    bool expected = false;
    if (!suspended_.compare_exchange_strong(expected, true)) return;

    submit_worker_->Pause();
    
    if (kPreemptMode >= kPreemptModeDeactivate) {
        hal_queue_->Deactivate();
    }

    if (kPreemptMode >= kPreemptModeInterrupt) {
        hal_queue_->Interrupt();
    }

    if (sync_hal_queue) {
        hal_queue_->HalSynchronize();
    }
}

void XQueue::Resume(bool drop_commands)
{
    // Will not resume if it is not suspended.
    bool expected = true;
    if (!suspended_.compare_exchange_strong(expected, false)) return;

    if (kPreemptMode == kPreemptModeStopSubmission) {
        // Should not clear the command log because when the xqueue
        // is suspended without hal synchronization, the hal commands
        // may be still running when resumes.
        submit_worker_->Resume();
        return;
    }

    // hal synchronization should be done in HalQueue::Reactivate.
    
    if (!drop_commands) {
        // Regular reactivate and then resume if not dropping commands.
        hal_queue_->Reactivate(submit_worker_->GetCommandLog());
        submit_worker_->Resume();
        return;
    }

    // Reactivate the hal queue with an empty command log.
    CommandLog empty_log = {};
    hal_queue_->Reactivate(empty_log);

    // Drop all hal commands in the wait queue.
    wait_queue_->Drop();

    // drop_idx is the last hal command idx that needs to be dropped.
    // Here drop_idx is the idx of the last hal command submitted.
    int64_t drop_idx = next_hal_command_idx_.load() - 1;
    submit_worker_->ResumeAndDrop(drop_idx);
}

XQueueState XQueue::GetState()
{
    return wait_queue_->CheckState();
}

std::shared_ptr<XCommand> XQueue::EnqueueSynchronizeCommand()
{
    return wait_queue_->EnqueueSynchronizeCommand();
}

size_t XQueue::Clear(std::function<bool(std::shared_ptr<XCommand> hal_command)> remove_filter) {
    return wait_queue_->Clear(remove_filter);
}

size_t XQueue::GetSize() {
    return wait_queue_->GetSize();
}
