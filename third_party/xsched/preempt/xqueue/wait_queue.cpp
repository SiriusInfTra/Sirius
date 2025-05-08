#include "utils/xassert.h"
#include "sched/protocol/event.h"
#include "preempt/event/dispatcher.h"
#include "preempt/xqueue/wait_queue.h"

using namespace xsched::sched;
using namespace xsched::preempt;

WaitQueue::WaitQueue(XQueueHandle handle)
    : kHandle(handle)
{
    Enqueue(std::make_shared<XQueueSynchronizeCommand>());
    XASSERT(last_command_ != nullptr, "last_command_ should not be nullptr");
}

XQueueState WaitQueue::CheckState()
{
    std::unique_lock<std::mutex> lock(mutex_);
    return xqueue_state_;
}

std::shared_ptr<XCommand> WaitQueue::Dequeue()
{
    std::unique_lock<std::mutex> lock(mutex_);

    if (!commands_.empty()) {
        auto xcommand = commands_.front();
        commands_.pop_front();
        return xcommand;
    }

    switch (last_command_->kCommandType)
    {
    case kCommandTypeXQueueSynchronize:
    {
        if (xqueue_state_ == kQueueStateReady) {
            xqueue_state_ = kQueueStateIdle;
            auto e = std::make_unique<XQueueIdleEvent>(kHandle);
            g_event_dispatcher.Dispatch(std::move(e));
        }

        while (commands_.empty()) cond_var_.wait(lock);
        auto xcommand = commands_.front();
        commands_.pop_front();
        return xcommand;
    }

    case kCommandTypeIntervalSynchronize:
    {
        auto xcommand = std::make_shared<XQueueSynchronizeCommand>();
        xcommand->BeforeEnqueued();
        xcommand->SetState(kCommandStateEnqueued);
        last_command_ = xcommand;
        return xcommand;
    }
    
    default:
    {
        auto xcommand = std::make_shared<IntervalSynchronizeCommand>();
        xcommand->BeforeEnqueued();
        xcommand->SetState(kCommandStateEnqueued);
        last_command_ = xcommand;
        return xcommand;
    }

    }

    XASSERT(false, "should not reach here");
    return nullptr;
}

void WaitQueue::Enqueue(std::shared_ptr<XCommand> xcommand)
{
    XASSERT(xcommand != nullptr, "xcommand should not be nullptr");

    xcommand->BeforeEnqueued();
    xcommand->SetState(kCommandStateEnqueued);

    mutex_.lock();

    if (xcommand->kCommandType == kCommandTypeHAL
        && xqueue_state_ == kQueueStateIdle) {
        
        xqueue_state_ = kQueueStateReady;
        auto e = std::make_unique<XQueueReadyEvent>(kHandle);
        g_event_dispatcher.Dispatch(std::move(e));
    }

    last_command_ = xcommand;
    commands_.emplace_back(xcommand);

    mutex_.unlock();
    cond_var_.notify_all();
}

void WaitQueue::Drop()
{
    mutex_.lock();

    for (auto &xcommand : commands_) {
        xcommand->SetState(kCommandStateCompleted);
        xcommand->AfterCompleted();
    }
    commands_.clear();

    mutex_.unlock();
}

std::shared_ptr<XCommand> WaitQueue::EnqueueSynchronizeCommand()
{
    std::unique_lock<std::mutex> lock(mutex_);

    if (last_command_->kCommandType == kCommandTypeXQueueSynchronize) {
        return last_command_;
    }

    auto xcommand = std::make_shared<XQueueSynchronizeCommand>();
    xcommand->BeforeEnqueued();
    xcommand->SetState(kCommandStateEnqueued);
    last_command_ = xcommand;
    commands_.emplace_back(xcommand);

    lock.unlock();
    cond_var_.notify_all();
    return xcommand;
}

size_t WaitQueue::GetSize() {
    mutex_.lock();
    auto size = commands_.size();
    mutex_.unlock();
    return size;
}

size_t WaitQueue::Clear(std::function<bool(std::shared_ptr<XCommand> hal_command)> filter) {
    mutex_.lock();
    size_t remove = 0;
    for (auto it = commands_.begin(); it != commands_.end(); ) {
        if (filter(*it)) {
            it = commands_.erase(it);
            remove++;
        } else {
            it++;
        }
    }
    mutex_.unlock();
    return remove;
}