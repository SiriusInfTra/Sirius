#include <cstring>

#include "utils/xassert.h"
#include "sched/protocol/operation.h"

using namespace xsched::sched;
using namespace xsched::preempt;

std::unique_ptr<const Operation> Operation::CopyConstructor(const void *data)
{
    auto meta = (const OperationMeta *)data;
    switch (meta->type)
    {
    case kOperationTerminate:
        return std::make_unique<TerminateOperation>(data);
    case kOperationSched:
        return std::make_unique<SchedOperation>(data);
    default:
        XASSERT(false, "unknown operation type: %d", meta->type);
        return nullptr;
    }
}

SchedOperation::SchedOperation(const void *data)
{
    size_t size = ((const OperationData *)data)->size;
    OperationData *new_data = (OperationData *)malloc(size);
    memcpy(new_data, data, size);
    data_ = new_data;
}

SchedOperation::SchedOperation(const ProcessStatus &status)
{
    PID pid = status.pid;
    size_t running_cnt = status.running_xqueues.size();
    size_t suspended_cnt = status.suspended_xqueues.size();
    size_t xqueue_cnt = running_cnt + suspended_cnt;
    
    size_t size = offsetof(OperationData, handles)
                + xqueue_cnt * sizeof(XQueueHandle);
    OperationData *data = (OperationData *)malloc(size);
    data_ = data;

    data->meta.type = kOperationSched;
    data->meta.pid = pid;
    data->size = size;
    data->running_cnt = running_cnt;
    data->suspended_cnt = suspended_cnt;

    xqueue_cnt = 0;
    for (auto &handle : status.running_xqueues) {
        data->handles[xqueue_cnt++] = handle;
    }
    for (auto &handle : status.suspended_xqueues) {
        data->handles[xqueue_cnt++] = handle;
    }
}

SchedOperation::~SchedOperation()
{
    if (data_) free((void *)data_);
}

