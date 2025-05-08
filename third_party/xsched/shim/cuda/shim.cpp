#include <list>
#include <mutex>
#include <memory>
#include <unordered_map>
#include <functional>

#include "utils/xassert.h"
#include "sched/policy/hpf.h"
#include "sched/policy/cbs.h"
#include "shim/cuda/shim.h"
#include "shim/cuda/xctrl.h"
#include "shim/cuda/handle.h"
#include "shim/common/agent.h"
#include "hal/cuda/cuda_queue.h"
#include "hal/cuda/cuda_command.h"
#include "preempt/xqueue/xqueue.h"

#include "shim/cuda/extra.h"

using namespace xsched::preempt;
using namespace xsched::hal::cuda;
using namespace xsched::shim::cuda;

static std::mutex blocking_xqueue_mutex;
static std::unordered_map<XQueueHandle, std::shared_ptr<XQueue>> blocking_xqueues;

static std::mutex xevent_mutex;
static std::unordered_map<CUevent, std::shared_ptr<CudaEventRecordCommand>> xevents;

static std::atomic<bool> cuda_xqueue_reject_calls{false};


CUDA_CTRL_FUNC uint64_t CudaXQueueCreate(CUstream stream,
                                         int preempt_mode,
                                         int64_t queue_length,
                                         int64_t sync_interval)
{
    XASSERT(stream != nullptr, "cannot enable xpreempt on default stream");
    unsigned int stream_flags;
    XASSERT(Driver::StreamGetFlags(stream, &stream_flags) == CUDA_SUCCESS,
            "fail to get Cuda stream flags");
    XQueueHandle handle = GetXHandle(stream);

    if (stream_flags == CU_STREAM_DEFAULT) blocking_xqueue_mutex.lock();

    auto cuda_queue = std::make_shared<CudaQueue>((XPreemptMode)preempt_mode, stream);
    auto xqueue = xsched::shim::XManager::CreateXQueue(cuda_queue, handle,
        kDeviceCUDA, (XPreemptMode)preempt_mode, queue_length, sync_interval);

    if (stream_flags == CU_STREAM_DEFAULT) {
        blocking_xqueues[handle] = xqueue;
        blocking_xqueue_mutex.unlock();
    }

    return (uint64_t)handle;
}

CUDA_CTRL_FUNC void CudaXQueueDestroy(uint64_t handle)
{
    blocking_xqueue_mutex.lock();
    xevent_mutex.lock();
    
    xsched::shim::XManager::DestroyXQueue(handle);
    blocking_xqueues.erase(handle);

    for (auto it = xevents.begin(); it != xevents.end();) {
        auto xqueue = it->second->GetXQueue();
        // xqueue could be nullptr because the xevent may not be
        // submitted to an XQueue
        if (xqueue && xqueue->GetXQueueHandle() != handle) ++it;
        else it = xevents.erase(it);
    }

    xevent_mutex.unlock();
    blocking_xqueue_mutex.unlock();
}

CUDA_CTRL_FUNC XQueue* CudaXQueueGet(uint64_t handle) {
    return xsched::shim::XManager::GetXQueue(handle).get();
}

CUDA_CTRL_FUNC void CudaXQueuePreempt(CUstream stream, bool sync_hal_queue)
{
    XQueueHandle handle = GetXHandle(stream);
    xsched::shim::XManager::Suspend(handle, sync_hal_queue);
}

CUDA_CTRL_FUNC void CudaXQueueResume(CUstream stream, bool drop_commands)
{
    XQueueHandle handle = GetXHandle(stream);
    xsched::shim::XManager::Resume(handle, drop_commands);
}

CUDA_CTRL_FUNC void CudaSyncBlockingXQueues()
{
    std::list<std::shared_ptr<XCommand>> sync_commands;

    blocking_xqueue_mutex.lock();
    for (auto it : blocking_xqueues) {
        sync_commands.emplace_back(it.second->EnqueueSynchronizeCommand());
    }
    blocking_xqueue_mutex.unlock();

    for (auto sync_command : sync_commands) {
        sync_command->Synchronize();
    }
}

CUDA_CTRL_FUNC void CudaXQueueSetPriority(CUstream stream, Prio priority)
{
    XQueueHandle handle = GetXHandle(stream);
    auto hint = std::make_unique<xsched::sched::SetPriorityHint>(
        (xsched::sched::Prio)priority, handle);
    xsched::shim::XManager::GiveHint(std::move(hint));
}

CUDA_CTRL_FUNC void CudaXQueueSetBandwidth(CUstream stream, Bwidth bdw)
{
    XQueueHandle handle = GetXHandle(stream);
    auto hint = std::make_unique<xsched::sched::SetBandwidthHint>(
        (xsched::sched::Bwidth)bdw, handle);
    xsched::shim::XManager::GiveHint(std::move(hint));
}

CUDA_CTRL_FUNC void CudaXQueueSetReject(bool reject) {
    cuda_xqueue_reject_calls.store(reject, std::memory_order_relaxed);
}

CUDA_CTRL_FUNC bool CudaXQueueQueryReject() {
    return cuda_xqueue_reject_calls;
}

CUDA_CTRL_FUNC bool CudaXQueueSync(uint64_t handle) {
    return xsched::shim::XManager::Synchronize(handle);
}

CUDA_CTRL_FUNC void CudaGuessNcclBegin() {
    xsched::shim::cuda::extra::Nccl::Init();
    xsched::shim::cuda::extra::Nccl::GuessNcclBegin();
}

CUDA_CTRL_FUNC void CudaGuessNcclEnd() {
    xsched::shim::cuda::extra::Nccl::Init();
    xsched::shim::cuda::extra::Nccl::GuessNcclEnd();
}

CUDA_CTRL_FUNC bool CudaIsGuessNcclBegined() {
    xsched::shim::cuda::extra::Nccl::Init();
    return xsched::shim::cuda::extra::Nccl::IsGuessNcclBegined();
}

CUDA_CTRL_FUNC void CudaNcclSteamGet(std::vector<CUstream> &nccl_stream_ret) {
    xsched::shim::cuda::extra::Nccl::Init();
    nccl_stream_ret = xsched::shim::cuda::extra::Nccl::GetStreams();
}

/////////////////////////////////////////////////////////////////////

namespace xsched::shim::cuda
{

CUresult XLaunchKernel(CUfunction f,
                       unsigned int grid_dim_x,
                       unsigned int grid_dim_y,
                       unsigned int grid_dim_z,
                       unsigned int block_dim_x,
                       unsigned int block_dim_y,
                       unsigned int block_dim_z,
                       unsigned int shared_mem_bytes,
                       CUstream stream,
                       void **kernel_params,
                       void **extra)
{
    if (extra::Nccl::IsGuessingNccl()) {
        extra::Nccl::MaybeAddStream(f, stream);
    }

    if (stream == nullptr) {
        CudaSyncBlockingXQueues();
        return Driver::LaunchKernel(
            f, grid_dim_x, grid_dim_y, grid_dim_z,
            block_dim_x, block_dim_y, block_dim_z,
            shared_mem_bytes, stream, kernel_params, extra);
    }
    
    XQueueHandle handle = GetXHandle(stream);
    auto xqueue = XManager::GetXQueue(handle);
    auto kernel = std::make_shared<CudaKernelLaunchCommand>(
        f, grid_dim_x, grid_dim_y, grid_dim_z,
        block_dim_x, block_dim_y, block_dim_z,
        shared_mem_bytes, kernel_params, extra, xqueue != nullptr);

    // if (std::string(extra::GetFuncName(f)).find("ncclKernel") != std::string::npos) {
    //     XINFO("%s %p on stream %p", extra::GetFuncName(f), kernel.get(), stream);
    // }

    if (xqueue == nullptr) {
        return CudaQueue::LaunchKernelNormal(kernel, stream);
    }

    xqueue->Submit(kernel);
    return CUDA_SUCCESS;
}

CUresult XEventQuery(CUevent event)
{
    if (event == nullptr) {
        return Driver::EventQuery(event);
    }
    
    xevent_mutex.lock();
    auto it = xevents.find(event);
    
    if (it == xevents.end()) {
        xevent_mutex.unlock();
        return Driver::EventQuery(event);
    }

    auto xevent = it->second;
    xevent_mutex.unlock();

    auto state = xevent->GetState();
    if (state == kCommandStateCompleted) return CUDA_SUCCESS;
    return CUDA_ERROR_NOT_READY;
}

CUresult XEventRecord(CUevent event, CUstream stream)
{
    if (event == nullptr) {
        return Driver::EventRecord(event, stream);
    }

    CUresult result;
    auto command = std::make_shared<CudaEventRecordCommand>(event);

    if (stream == nullptr) {
        CudaSyncBlockingXQueues();
        result = Driver::EventRecord(event, stream);
    } else {
        result = SubmitCudaCommand(command, stream);
    }

    xevent_mutex.lock();
    xevents[event] = command;
    xevent_mutex.unlock();

    return result;
}

CUresult XEventRecordWithFlags(CUevent event, CUstream stream,
                               unsigned int flags)
{
    if (event == nullptr) {
        return Driver::EventRecord(event, stream);
    }

    CUresult result;
    auto command = std::make_shared<CudaEventRecordWithFlagsCommand>(event, flags);

    if (stream == nullptr) {
        CudaSyncBlockingXQueues();
        result = Driver::EventRecord(event, stream);
    } else {
        result = SubmitCudaCommand(command, stream);
    }

    xevent_mutex.lock();
    xevents[event] = command;
    xevent_mutex.unlock();

    return result;
}

CUresult XEventSynchronize(CUevent event)
{
    if (event == nullptr) {
        return Driver::EventSynchronize(event);
    }
    
    xevent_mutex.lock();
    auto it = xevents.find(event);

    if (it == xevents.end()) {
        xevent_mutex.unlock();
        return Driver::EventSynchronize(event);
    }

    auto xevent = it->second;
    xevent_mutex.unlock();

    xevent->Synchronize();
    return CUDA_SUCCESS;
}

CUresult XStreamWaitEvent(CUstream stream, CUevent event, unsigned int flags)
{
    if (event == nullptr) {
        return Driver::StreamWaitEvent(stream, event, flags);
    }

    xevent_mutex.lock();
    auto xevent_it = xevents.find(event);

    if (xevent_it == xevents.end()) {
        // the event is not recorded yet
        xevent_mutex.unlock();
        return Driver::StreamWaitEvent(stream, event, flags);
    }

    std::shared_ptr<CudaEventRecordCommand> xevent = xevent_it->second;
    xevent_mutex.unlock();

    if (stream == nullptr) {
        // sync a event on default stream
        CudaSyncBlockingXQueues();
        xevent->Synchronize();
        return Driver::StreamWaitEvent(stream, event, flags);
    }

    XQueueHandle handle = GetXHandle(stream);
    auto xqueue = XManager::GetXQueue(handle);

    if (xqueue == nullptr) {
        // waiting stream is not a xqueue
        if (xevent->GetXQueue() == nullptr) {
            // the event is not recorded on a xqueue
            return Driver::StreamWaitEvent(stream, event, flags);
        }

        xevent->Synchronize();
        return CUDA_SUCCESS;
    }

    auto command = std::make_shared<CudaEventWaitCommand>(xevent, flags);
    xqueue->Submit(command);
    return CUDA_SUCCESS;
}

CUresult XEventDestroy(CUevent event)
{
    if (event == nullptr) {
        return Driver::EventDestroy(event);
    }
    
    xevent_mutex.lock();
    auto it = xevents.find(event);
    if (it == xevents.end()) {
        xevent_mutex.unlock();
        return Driver::EventDestroy(event);
    }

    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g593ec73a8ec5a5fc031311d3e4dca1ef
    // According to CUDA driver API documentation, if the event is waiting
    // in XQueues, we should not destroy it immediately. Instead, we shall
    // set a flag to destroy the CUevent in the destructor of the xevent.
    it->second->DestroyEvent();
    xevents.erase(it);
    xevent_mutex.unlock();
    return CUDA_SUCCESS;
}

CUresult XEventDestroy_v2(CUevent event)
{
    if (event == nullptr) {
        return Driver::EventDestroyV2(event);
    }
    
    xevent_mutex.lock();
    auto it = xevents.find(event);
    if (it == xevents.end()) {
        xevent_mutex.unlock();
        return Driver::EventDestroyV2(event);
    }

    // Same as XEventDestroy.
    it->second->DestroyEvent();
    xevents.erase(it);
    xevent_mutex.unlock();
    return CUDA_SUCCESS;
}

CUresult XStreamSynchronize(CUstream stream)
{
    if (XManager::Synchronize(GetXHandle(stream))) {
        return CUDA_SUCCESS;
    }
    return Driver::StreamSynchronize(stream);
}

CUresult XStreamQuery(CUstream stream)
{
    switch (XManager::GetXQueueState(GetXHandle(stream)))
    {
    case kQueueStateIdle:
        return CUDA_SUCCESS;
    case kQueueStateReady:
        return CUDA_ERROR_NOT_READY;
    default:
        return Driver::StreamQuery(stream);
    }
}
CUresult XCtxSynchronize()
{
    XManager::SynchronizeAllXQueues();
    return Driver::CtxSynchronize();
}

CUresult XStreamCreate(CUstream *stream, unsigned int flags)
{
    return Driver::StreamCreate(stream, flags);
}

CUresult XStreamCreateWithPriority(CUstream *stream, unsigned int flags,
                                   int priority)
{
    return Driver::StreamCreateWithPriority(stream, flags, priority);
}

} // namespace xsched::shim::cuda
