#pragma once

#include <cstddef>
#include <cstdint>

#include "sched/protocol/status.h"

namespace xsched::sched
{

enum OperationType
{
    kOperationTerminate     = 0,
    kOperationSched         = 1,
};

struct OperationMeta
{
    OperationType type;
    PID           pid;
};

class Operation
{
public:
    Operation() = default;
    virtual ~Operation() = default;

    /// @brief Get the data of the operation. MUST start with OperationMeta.
    virtual const void *  Data() const = 0;
    virtual size_t        Size() const = 0;
    virtual OperationType Type() const = 0;
    virtual PID           Pid()  const = 0;

    static std::unique_ptr<const Operation> CopyConstructor(const void *data);
};

/// @brief Terminate the worker thread of the scheduler.
class TerminateOperation : public Operation
{
public:
    TerminateOperation()
        : meta_{ .type = kOperationTerminate, .pid = GetProcessId() } {}
    TerminateOperation(const void *data)
        : meta_(*(const OperationMeta *)data) {}
    
    virtual ~TerminateOperation() = default;

    virtual const void *  Data() const override { return (void *)&meta_; }
    virtual size_t        Size() const override { return sizeof(meta_); }
    virtual OperationType Type() const override { return kOperationTerminate; }
    virtual PID           Pid()  const override { return meta_.pid; }

private:
    const OperationMeta meta_;
};

class SchedOperation : public Operation
{
public:
    SchedOperation(const void *data);
    SchedOperation(const ProcessStatus &status);
    virtual ~SchedOperation();

    virtual const void *  Data() const override { return data_; }
    virtual size_t        Size() const override { return data_->size; }
    virtual OperationType Type() const override { return kOperationSched; }
    virtual PID           Pid()  const override { return data_->meta.pid; }

    size_t RunningCnt() const { return data_->running_cnt; }
    size_t SuspendedCnt() const { return data_->suspended_cnt; }
    const preempt::XQueueHandle *Handles() const { return data_->handles; }

private:
    struct OperationData
    {
        OperationMeta meta;
        /// @brief The size of the whole OperationData.
        size_t        size;
        size_t        running_cnt;
        size_t        suspended_cnt;
        /// @brief 64 is only a placeholder.
        /// The actual number of handles is (running_cnt + suspended_cnt).
        /// The handles are first running_cnt running xqueues, then 
        /// suspended_cnt suspended xqueues.
        preempt::XQueueHandle handles[64];
    };

    const OperationData *data_ = nullptr;
};

} // namespace xsched::sched
