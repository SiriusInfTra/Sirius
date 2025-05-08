#include "utils/xassert.h"
#include "preempt/xqueue/worker.h"

using namespace xsched::utils;
using namespace xsched::preempt;

Worker::Worker(std::shared_ptr<HalQueue> hal_queue,
               std::shared_ptr<WaitQueue> wait_queue,
               XPreemptMode mode,
               int64_t queue_length,
               int64_t sync_interval)
    : kPreemptMode(mode)
    , kSyncInterval(sync_interval)
    , kQueueLength(queue_length)
    , hal_queue_(hal_queue)
    , wait_queue_(wait_queue)
    , mutex_(std::make_unique<MCSLock>())
{
    // Queue length must not be smaller than sync interval.
    // Otherwise, there could be no commands to wait if the queue is full.
    XASSERT(queue_length >= sync_interval,
            "queue_length must not be smaller than sync_interval");

    worker_thread_ = std::make_unique<std::thread>([=](){
        // Initialize hal_queue on the submitting thread. Hal can call
        // platform specific APIs like cuCtxSetCurrent() in Initialize().
        hal_queue_->OnInitialize();
        this->WorkerLoop();
    });
}

Worker::~Worker()
{
    // When destroying the XQueue holding this Worker, the XQueue will
    // enqueue an XQueueDestroyCommand. When the worker_thread_ consumes
    // the XQueueDestroyCommand, it will call SynchronizeQueueInternal
    // and wait for all hal commands to complete, and then clear data
    // and exit. ~Worker() should join the submit thread.
    worker_thread_->join();
}

void Worker::Pause()
{
    mutex_->lock();
    state_ = kWorkerStatePaused;
    pause_count_ += 1;
    mutex_->unlock();
}

void Worker::Resume()
{
    mutex_->lock();
    state_ = kWorkerStateRunning;
    mutex_->unlock();
    cond_var_.notify_all();
}

void Worker::ResumeAndDrop(int64_t drop_idx)
{
    std::unique_lock<MutexLock> lock(*mutex_);

    // Clear all hal commands.
    sync_command_log_.clear();

    for (auto hal_command : command_log_) {
        hal_command->SetState(kCommandStateCompleted);
        hal_command->AfterCompleted();
    }
    command_log_.clear();

    drop_idx_ = drop_idx;
    state_ = kWorkerStateRunning;
    
    lock.unlock();
    cond_var_.notify_all();
}

void Worker::SynchronizeQueue()
{
    std::unique_lock<MutexLock> lock(*mutex_);
    SynchronizeQueueInternal(std::move(lock));
}

void Worker::SynchronizeCommand(std::shared_ptr<HalCommand> hal_command)
{
    std::unique_lock<MutexLock> lock(*mutex_);
    SynchronizeCommandInternal(hal_command, std::move(lock));
}

void Worker::RegisterAfterHalSubmitHook(
    std::function<void(std::shared_ptr<HalCommand>)> fn)
{
    after_hal_submit_hook_ = fn;
}

void Worker::WorkerLoop()
{
    while (true) {
        std::shared_ptr<XCommand> xcommand = wait_queue_->Dequeue();

        switch (xcommand->kCommandType)
        {
        case kCommandTypeHAL:
        {
            auto hal_command = std::static_pointer_cast<HalCommand>(xcommand);
            SubmitHalCommand(hal_command);
            break;
        }

        case kCommandTypeXQueueSynchronize:
        {
            xcommand->SetState(kCommandStateHalSubmmited);
            SynchronizeQueue();
            xcommand->SetState(kCommandStateCompleted);
            xcommand->AfterCompleted();
            break;
        }

        case kCommandTypeIntervalSynchronize:
        {
            std::unique_lock<MutexLock> lock(*mutex_);

            if (sync_command_log_.empty()) break;
            auto command = sync_command_log_.front();
            lock = SynchronizeCommandInternal(command, std::move(lock));
            lock.unlock();

            xcommand->SetState(kCommandStateCompleted);
            xcommand->AfterCompleted();
            break;
        }
        
        case kCommandTypeXQueueDestroy:
        {
            std::unique_lock<MutexLock> lock(*mutex_);
            lock = SynchronizeQueueInternal(std::move(lock));
            state_ = kWorkerStateTerminated;
            
            lock.unlock();
            cond_var_.notify_all();

            xcommand->SetState(kCommandStateCompleted);
            xcommand->AfterCompleted();
            // Exit the worker thread.
            return;
        }
        
        default:
            XASSERT(false, "unknown command type: %d", xcommand->kCommandType);
            break;
        }
    }
}

void Worker::SubmitHalCommand(std::shared_ptr<HalCommand> hal_command)
{
    // If a HalCommand is deactivated, all non-idempotent HalCommands submitted
    // after it should also be deactivated. Otherwise the XPU will become 
    // inconsist. So, if a HalCommand is not deactivatable, it should wait
    // until all submitted deactivatable HalCommands finish.
    // This is unnecessary under kPreemptModeStopSubmission.
    bool wait_deactivatable =
        kPreemptMode >= kPreemptModeDeactivate &&
        hal_command->kHalCommandType == kHalCommandTypeNormal;

    hal_command->BeforeHalSubmit();
    std::unique_lock<MutexLock> lock(*mutex_);

    // If hal command is not deactivatable and non-idempoent,
    // wait until all previous deactivatable commands are completed.
    if (wait_deactivatable) {
        // Find the last synchronizable command after
        // the last cancelable command.
        bool has_deactivatable = false;
        std::shared_ptr<HalCommand> command_to_sync = nullptr;
        for (auto it = command_log_.rbegin();
                it != command_log_.rend(); ++it) {
            if ((*it)->HalSynchronizable()) command_to_sync = *it;
            if ((*it)->kHalCommandType == kHalCommandTypeDeactivatable) {
                has_deactivatable = true;
                break;
            }
        }

        if (has_deactivatable) {
            // If there is a synchronizable command, sync it.
            // Otherwise, sync the hal queue.
            lock = (command_to_sync == nullptr)
                 ? SynchronizeQueueInternal(std::move(lock))
                 : SynchronizeCommandInternal(command_to_sync, std::move(lock));
        }
    }

    if (command_log_.size() >= (size_t)kQueueLength) {
        // The command log is full, wait for a empty slot.
        std::shared_ptr<HalCommand> command_to_sync = nullptr;
        int64_t front_command_idx = command_log_.front()->GetIdx();

        for (auto cmd : sync_command_log_) {
            // Make sure after syncing the command,
            // there will be at least one empty slot.
            if (cmd->GetIdx() >= front_command_idx) {
                command_to_sync = cmd;
                break;
            }
        }

        // If there is a synchronizable command, sync it.
        // Otherwise, sync the hal queue.
        lock = (command_to_sync == nullptr)
             ? SynchronizeQueueInternal(std::move(lock))
             : SynchronizeCommandInternal(command_to_sync, std::move(lock));

    } else {
        // Wait if the worker is paused.
        while (true) {
            if (state_ == kWorkerStateRunning) {
                break;
            } else if (state_ == kWorkerStatePaused) {
                cond_var_.wait(lock);
            } else if (state_ == kWorkerStateTerminated) {
                // state_ will not be kWorkerStateTerminated,
                // because SubmitHalCommand() will only be called in
                // worker thread. Right after state_ is set to
                // kWorkerStateTerminated, the thread will then exit.
                XASSERT(false,
                        "Worker state should not be kWorkerStateTerminated.");
            } else {
                XASSERT(false, "Invalid worker state");
            }
        }
    }

    // Check if the command should be dropped.
    if (hal_command->GetIdx() <= drop_idx_) {
        hal_command->SetState(kCommandStateCompleted);
        hal_command->AfterCompleted();
        return;
    }

    // Reach sync interval, should enable hal synchronization for the command.
    if (hal_command->GetIdx() - last_synchronizable_idx_ >= kSyncInterval) {
        hal_command->EnableHalSynchronization();
    }

    // Submit the command.
    hal_queue_->HalSubmit(hal_command);
    hal_command->SetState(kCommandStateHalSubmmited);

    if (after_hal_submit_hook_ != nullptr) {
        after_hal_submit_hook_(hal_command);
    }

    command_log_.emplace_back(hal_command);

    if (hal_command->HalSynchronizable()) {
        sync_command_log_.emplace_back(hal_command);
        last_synchronizable_idx_ = hal_command->GetIdx();
    }
}

std::unique_lock<MutexLock>
Worker::SynchronizeQueueInternal(std::unique_lock<MutexLock> lock)
{
    while (true) {
        // Wait if the worker is paused.
        while (true) {
            if (state_ == kWorkerStateRunning) {
                break;
            } else if (state_ == kWorkerStatePaused) {
                cond_var_.wait(lock);
            } else if (state_ == kWorkerStateTerminated) {
                return lock;
            } else {
                XASSERT(false, "Invalid worker state");
            }
        }

        // If there is no hal command submitted, then no need to sync.
        if (command_log_.size() == 0) return lock;

        int64_t current_pause_cnt = pause_count_;

        lock.unlock();
        hal_queue_->HalSynchronize();
        lock.lock();

        // Check if preemption happened during hal_queue_->HalSynchronize().
        if (current_pause_cnt == pause_count_) break;
    }

    // Pop and delete all hal commands submitted.
    sync_command_log_.clear();

    for (auto hal_command : command_log_) {
        hal_command->SetState(kCommandStateCompleted);
        hal_command->AfterCompleted();
    }
    command_log_.clear();

    return lock;
}

std::unique_lock<MutexLock>
Worker::SynchronizeCommandInternal(std::shared_ptr<HalCommand> hal_command,
                                   std::unique_lock<MutexLock> lock)
{
    XASSERT(hal_command->HalSynchronizable(),
    "The hal command should be synchronizable");

    while (true) {
        // Wait if the worker is paused.
        while (true) {
            if (state_ == kWorkerStateRunning) {
                break;
            } else if (state_ == kWorkerStatePaused) {
                cond_var_.wait(lock);
            } else if (state_ == kWorkerStateTerminated) {
                return lock;
            } else {
                XASSERT(false, "Invalid worker state");
            }
        }

        XCommandState state = hal_command->GetState();
        XASSERT(state >= kCommandStateHalSubmmited,
                "Sync command should be submitted first");
        if (state == kCommandStateCompleted) break;

        int64_t current_pause_cnt = pause_count_;

        lock.unlock();
        hal_command->HalSynchronize();
        lock.lock();

        // Check if preemption happened during hal_queue_->HalSynchronize().
        if (current_pause_cnt == pause_count_) break;
    }

    // Pop and delete all hal commands submitted in previous.
    const int64_t current_command_idx = hal_command->GetIdx();

    while (sync_command_log_.size() > 0) {
        if (sync_command_log_.front()->GetIdx()
                > current_command_idx) break;
        sync_command_log_.pop_front();
    }

    while (command_log_.size() > 0) {
        auto front_command = command_log_.front();
        if (front_command->GetIdx() > current_command_idx) break;

        front_command->SetState(kCommandStateCompleted);
        front_command->AfterCompleted();
        command_log_.pop_front();
    }

    return lock;
}
