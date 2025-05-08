#pragma once

#include <memory>
#include <atomic>
#include <functional>

#include "preempt/hal/hal_queue.h"
#include "preempt/hal/hal_command.h"

#include "preempt/xqueue/xtype.h"
#include "preempt/xqueue/worker.h"
#include "preempt/xqueue/wait_queue.h"

namespace xsched::preempt
{

class XQueue : public std::enable_shared_from_this<XQueue>
{
public:
    XQueue(std::shared_ptr<HalQueue> hal_queue,
           XQueueHandle handle,
           XDevice device,
           XPreemptMode mode,
           int64_t queue_length,
           int64_t sync_interval);
    ~XQueue();

    void Submit(std::shared_ptr<HalCommand> hal_command);

    void Synchronize();
    void Synchronize(std::shared_ptr<HalCommand> hal_command);

    void Suspend(bool sync_hal_queue);
    void Resume(bool drop_commands);

    XQueueState GetState();
    XQueueHandle GetXQueueHandle() const { return kHandle; }
    std::shared_ptr<XCommand> EnqueueSynchronizeCommand();

    Worker* GetSubmitWorker() { return submit_worker_.get(); }

    size_t Clear(std::function<bool(std::shared_ptr<XCommand> hal_command)> remove_filter);
    size_t GetSize();

private:
    const XQueueHandle kHandle;
    const XDevice      kDevice;
    const XPreemptMode kPreemptMode;

    const std::shared_ptr<HalQueue> hal_queue_;
    const std::shared_ptr<WaitQueue> wait_queue_;
    const std::shared_ptr<Worker> submit_worker_;

    std::atomic_bool suspended_ = { false };
    std::atomic_bool terminated_ = { false };

    // The index of HalCommand starts from 1.
    std::atomic<int64_t> next_hal_command_idx_ = { 1 };
};

} // namespace xsched::preempt
