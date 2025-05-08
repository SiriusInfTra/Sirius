#include "preempt/xqueue/xcommand.h"

using namespace xsched::preempt;

XCommandState XCommand::GetState()
{
    state_mutex_.lock();
    XCommandState state = state_;
    state_mutex_.unlock();
    return state;
}

void XCommand::SetState(XCommandState state)
{
    state_mutex_.lock();
    state_ = state;
    state_mutex_.unlock();
    state_cond_var_.notify_all();
}

void XCommand::WaitUntil(XCommandState state)
{
    std::unique_lock<std::mutex> lock(state_mutex_);
    while ((int)state_ < (int)state) state_cond_var_.wait(lock);
}

void XCommand::Synchronize()
{
    WaitUntil(kCommandStateCompleted);
}

void XCommand::BeforeEnqueued()
{
    // nothing to do, just for override
}

void XCommand::AfterCompleted()
{
    // nothing to do, just for override
}
