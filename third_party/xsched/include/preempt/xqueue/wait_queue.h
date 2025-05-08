#pragma once

#include <list>
#include <mutex>
#include <memory>
#include <condition_variable>
#include <functional>

#include "preempt/xqueue/xtype.h"
#include "preempt/xqueue/xcommand.h"

namespace xsched::preempt
{

class WaitQueue
{
public:
    WaitQueue(XQueueHandle handle);
    ~WaitQueue() = default;

    XQueueState CheckState();

    /// @brief Dequeue an XCommand from the queue.
    /// @return Pointer to the dequeued XCommand.
    std::shared_ptr<XCommand> Dequeue();

    /// @brief Enqueue an XCommand to the queue.
    /// @param xcommand Pointer to the enqueued XCommand.
    void Enqueue(std::shared_ptr<XCommand> xcommand);

    void Drop();
    std::shared_ptr<XCommand> EnqueueSynchronizeCommand();

    size_t GetSize();
    size_t Clear(std::function<bool(std::shared_ptr<XCommand> hal_command)> filter);

private:
    const XQueueHandle kHandle;

    std::mutex mutex_;
    std::condition_variable cond_var_;

    XQueueState xqueue_state_ = kQueueStateIdle;
    std::shared_ptr<XCommand> last_command_ = nullptr;
    std::list<std::shared_ptr<XCommand>> commands_;
};

} // namespace xsched::preempt
