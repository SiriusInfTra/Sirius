#pragma once

#include "utils/common.h"
#include "sched/protocol/hint.h"
#include "preempt/xqueue/xtype.h"
#include "preempt/xqueue/xqueue.h"
#include "preempt/hal/hal_queue.h"

namespace xsched::shim
{

class XManager
{
public:
    STATIC_CLASS(XManager);

    static std::shared_ptr<preempt::XQueue>
    CreateXQueue(std::shared_ptr<preempt::HalQueue> hal_queue,
                 preempt::XQueueHandle handle,
                 preempt::XDevice device,
                 preempt::XPreemptMode mode,
                 int64_t queue_length,
                 int64_t sync_interval);

    static void DestroyXQueue(preempt::XQueueHandle handle);

    static std::shared_ptr<preempt::XQueue> GetXQueue(
        preempt::XQueueHandle handle);

    /// @brief 
    /// @param hal_command 
    /// @param handle 
    /// @return Submit successfully or not.
    static bool Submit(std::shared_ptr<preempt::HalCommand> hal_command,
                       preempt::XQueueHandle handle);
    
    /// @brief 
    /// @param handle 
    /// @return Synchronize successfully or not.
    static bool Synchronize(preempt::XQueueHandle handle);
    static void SynchronizeAllXQueues();

    static void Suspend(preempt::XQueueHandle handle, bool sync_hal_queue);
    static void Resume(preempt::XQueueHandle handle, bool drop_commands);

    static preempt::XQueueState GetXQueueState(preempt::XQueueHandle handle);

    static void GiveHint(std::unique_ptr<const sched::Hint> hint);
};

} // namespace xsched::shim
