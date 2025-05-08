#include <cstring>

#include "utils/xassert.h"
#include "preempt/xqueue/xqueue.h"
#include "hal/cudla/event_pool.h"
#include "hal/cudla/cudla_assert.h"
#include "hal/cudla/cudla_command.h"

using namespace xsched::hal::cudla;

CudlaCommand::~CudlaCommand()
{
    if (following_event_ == nullptr) return;
    g_event_pool.Push(following_event_);
}

cudaError_t CudlaCommand::EnqueueWrapper(cudaStream_t stream)
{
    cudaError_t ret = Enqueue(stream);
    if (UNLIKELY(ret != cudaSuccess)) return ret;
    if (following_event_ != nullptr) {
        ret = RtDriver::EventRecord(following_event_, stream);
    }
    return ret;
}

void CudlaCommand::HalSynchronize()
{
    XASSERT(following_event_ != nullptr, "following_event_ is nullptr");
    CUDART_ASSERT(RtDriver::EventSynchronize(following_event_));
}

bool CudlaCommand::HalSynchronizable()
{
    return following_event_ != nullptr;
}

bool CudlaCommand::EnableHalSynchronization()
{
    following_event_ = (cudaEvent_t)g_event_pool.Pop();
    return following_event_ != nullptr;
}

CudlaTaskCommand::CudlaTaskCommand(cudlaDevHandle const dev_handle,
                                   const cudlaTask * const tasks,
                                   uint32_t const num_tasks,
                                   uint32_t const flags)
    : CudlaCommand(preempt::kHalCommandTypeNormal)
    , dev_handle_(dev_handle)
    , num_tasks_(num_tasks)
    , flags_(flags)
{
    XASSERT(tasks != nullptr, "tasks should not be nullptr");
    tasks_ = (cudlaTask *)malloc(sizeof(cudlaTask) * num_tasks);
    memcpy(tasks_, tasks, sizeof(cudlaTask) * num_tasks);
}

CudlaTaskCommand::~CudlaTaskCommand()
{
    free(tasks_);
}

cudaError_t CudlaTaskCommand::Enqueue(cudaStream_t stream)
{
    CUDLA_ASSERT(DlaDriver::SubmitTask(
        dev_handle_, tasks_, num_tasks_, stream, flags_));
    return cudaSuccess;
}

CudlaMemoryCommand::CudlaMemoryCommand(void *dst, const void *src,
                                       size_t size, cudaMemcpyKind kind)
    : CudlaCommand(preempt::kHalCommandTypeNormal)
    , dst_(dst), src_(src), size_(size), kind_(kind)
{
    XASSERT(dst_ != nullptr, "dst_ should not be nullptr");
    XASSERT(src_ != nullptr, "src_ should not be nullptr");
}

cudaError_t CudlaMemoryCommand::Enqueue(cudaStream_t stream)
{
    return RtDriver::MemcpyAsync(dst_, src_, size_, kind_, stream);
}

CudlaEventRecordCommand::CudlaEventRecordCommand(cudaEvent_t event)
    : CudlaCommand(preempt::kHalCommandTypeIdempotent), event_(event)
{
    XASSERT(event_ != nullptr, "cuda event should not be nullptr");
}

void CudlaEventRecordCommand::HalSynchronize()
{
    CUDART_ASSERT(RtDriver::EventSynchronize(event_));
}

cudaError_t CudlaEventRecordCommand::Enqueue(cudaStream_t stream)
{
    return RtDriver::EventRecord(event_, stream);
}
