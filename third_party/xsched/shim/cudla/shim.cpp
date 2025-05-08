#include <mutex>
#include <unordered_map>

#include "utils/xassert.h"
#include "sched/policy/hpf.h"
#include "sched/policy/cbs.h"
#include "shim/cudla/shim.h"
#include "shim/cudla/xctrl.h"
#include "shim/cudla/handle.h"
#include "shim/common/agent.h"
#include "shim/common/xmanager.h"
#include "preempt/xqueue/xqueue.h"
#include "hal/cudla/cudla_queue.h"
#include "hal/cudla/cudla_assert.h"
#include "hal/cudla/cudla_command.h"

using namespace xsched::preempt;
using namespace xsched::hal::cudla;
using namespace xsched::shim::cudla;

static std::mutex xevent_mutex;
static std::unordered_map<cudaEvent_t,
                          std::shared_ptr<CudlaEventRecordCommand>> xevents;

CUDLA_CTRL_FUNC uint64_t CudlaXQueueCreate(cudaStream_t stream,
                                           int preempt_mode,
                                           int64_t queue_length,
                                           int64_t sync_interval)
{
    XASSERT(stream != nullptr, "cannot enable xpreempt on default stream");
    XASSERT(preempt_mode <= kPreemptModeStopSubmission,
            "only kXPreemptModeStopSubmission is supported");
    
    XQueueHandle handle = GetXHandle(stream);
    XPreemptMode mode = (XPreemptMode)preempt_mode;
    auto cudla_queue = std::make_shared<CudlaQueue>(mode, stream);
    auto xqueue = xsched::shim::XManager::CreateXQueue(cudla_queue, handle,
        kDeviceCUDLA, mode, queue_length, sync_interval);
    
    return (uint64_t)handle;
}

CUDLA_CTRL_FUNC void CudlaXQueueDestroy(uint64_t handle)
{
    xsched::shim::XManager::DestroyXQueue(handle);
}

CUDLA_CTRL_FUNC void CudlaXQueueSuspend(cudaStream_t stream,
                                        bool sync_hal_queue)
{
    XQueueHandle handle = GetXHandle(stream);
    xsched::shim::XManager::Suspend(handle, sync_hal_queue);
}

CUDLA_CTRL_FUNC void CudlaXQueueResume(cudaStream_t stream,
                                       bool drop_commands)
{
    XQueueHandle handle = GetXHandle(stream);
    xsched::shim::XManager::Resume(handle, drop_commands);
}

CUDLA_CTRL_FUNC void CudlaXQueueSetPriority(cudaStream_t stream, Prio prio)
{
    XQueueHandle handle = GetXHandle(stream);
    auto hint = std::make_unique<xsched::sched::SetPriorityHint>(
        (xsched::sched::Prio)prio, handle);
    xsched::shim::XManager::GiveHint(std::move(hint));
}

CUDLA_CTRL_FUNC void CudlaXQueueSetBandwidth(cudaStream_t stream, Bwidth bdw)
{
    XQueueHandle handle = GetXHandle(stream);
    auto hint = std::make_unique<xsched::sched::SetBandwidthHint>(
        (xsched::sched::Bwidth)bdw, handle);
    xsched::shim::XManager::GiveHint(std::move(hint));
}

namespace xsched::shim::cudla
{

cudaError_t XStreamCreate(cudaStream_t *stream)
{
    // TODO: generate sched event
    return RtDriver::StreamCreate(stream);
}

cudaError_t XStreamCreateWithFlags(cudaStream_t *stream, unsigned int flags)
{
    // TODO: generate sched event
    return RtDriver::StreamCreateWithFlags(stream, flags);
}

cudaError_t XStreamDestroy(cudaStream_t stream)
{
    // TODO: generate sched event
    return RtDriver::StreamDestroy(stream);
}

cudaError_t XStreamSynchronize(cudaStream_t stream)
{
    if (XManager::Synchronize(GetXHandle(stream))) return cudaSuccess;
    return RtDriver::StreamSynchronize(stream);
}

cudaError_t XEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    if (event == nullptr || stream == nullptr) {
        return RtDriver::EventRecord(event, stream);
    }

    auto xevent = std::make_shared<CudlaEventRecordCommand>(event);
    cudaError_t ret = SubmitCudlaCommand(xevent, stream);
    if (UNLIKELY(ret != cudaSuccess)) return ret;
    
    xevent_mutex.lock();
    xevents[event] = xevent;
    xevent_mutex.unlock();

    return ret;
}

cudaError_t XEventSynchronize(cudaEvent_t event)
{
    if (event == nullptr) {
        return RtDriver::EventSynchronize(event);
    }

    xevent_mutex.lock();
    auto it = xevents.find(event);

    if (it == xevents.end()) {
        xevent_mutex.unlock();
        return RtDriver::EventSynchronize(event);
    }

    auto xevent = it->second;
    xevent_mutex.unlock();

    xevent->Synchronize();
    return cudaSuccess;
}

cudaError_t XEventDestroy(cudaEvent_t event)
{
    if (event == nullptr) {
        return RtDriver::EventDestroy(event);
    }

    xevent_mutex.lock();
    xevents.erase(event);
    xevent_mutex.unlock();

    return RtDriver::EventDestroy(event);
}

cudlaStatus XSubmitTask(cudlaDevHandle const dev_handle,
                        const cudlaTask * const tasks,
                        uint32_t const num_tasks,
                        void* const stream,
                        uint32_t const flags)
{
    auto cmd = std::make_shared<CudlaTaskCommand>(dev_handle, tasks,
                                                  num_tasks, flags);
    CUDART_ASSERT(SubmitCudlaCommand(cmd, (cudaStream_t)stream));
    return cudlaSuccess;
}

} // namespace xsched::shim::cudla
