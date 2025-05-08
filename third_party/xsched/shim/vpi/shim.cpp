#include <mutex>
#include <unordered_map>

#include "utils/xassert.h"
#include "sched/policy/hpf.h"
#include "sched/policy/cbs.h"
#include "shim/vpi/shim.h"
#include "shim/vpi/xctrl.h"
#include "shim/vpi/handle.h"
#include "shim/common/agent.h"
#include "shim/common/xmanager.h"
#include "preempt/xqueue/xqueue.h"
#include "hal/vpi/vpi_queue.h"
#include "hal/vpi/vpi_assert.h"
#include "hal/vpi/vpi_command.h"

using namespace xsched::preempt;
using namespace xsched::hal::vpi;
using namespace xsched::shim::vpi;

static std::mutex xevent_mutex;
static std::unordered_map<VPIEvent,
                          std::shared_ptr<VpiEventRecordCommand>> xevents;

VPI_CTRL_FUNC uint64_t VpiXQueueCreate(VPIStream stream,
                                       int preempt_mode,
                                       int64_t queue_length,
                                       int64_t sync_interval)
{
    XASSERT(stream != nullptr, "cannot enable xpreempt on default stream");
    XASSERT(preempt_mode <= kPreemptModeStopSubmission,
            "only kXPreemptModeStopSubmission is supported");
    
    XQueueHandle handle = GetXHandle(stream);
    XPreemptMode mode = (XPreemptMode)preempt_mode;
    auto vpi_queue = std::make_shared<VpiQueue>(mode, stream);
    auto xqueue = xsched::shim::XManager::CreateXQueue(vpi_queue, handle,
        kDeviceVPI, mode, queue_length, sync_interval);
    
    return (uint64_t)handle;
}

VPI_CTRL_FUNC void VpiXQueueDestroy(uint64_t handle)
{
    xsched::shim::XManager::DestroyXQueue(handle);
}

VPI_CTRL_FUNC void VpiXQueueSuspend(VPIStream stream, bool sync_hal_queue)
{
    XQueueHandle handle = GetXHandle(stream);
    xsched::shim::XManager::Suspend(handle, sync_hal_queue);
}

VPI_CTRL_FUNC void VpiXQueueResume(VPIStream stream, bool drop_commands)
{
    XQueueHandle handle = GetXHandle(stream);
    xsched::shim::XManager::Resume(handle, drop_commands);
}

VPI_CTRL_FUNC void VpiXQueueSetPriority(VPIStream stream, Prio prio)
{
    XQueueHandle handle = GetXHandle(stream);
    auto hint = std::make_unique<xsched::sched::SetPriorityHint>(
        (xsched::sched::Prio)prio, handle);
    xsched::shim::XManager::GiveHint(std::move(hint));
}

VPI_CTRL_FUNC void VpiXQueueSetBandwidth(VPIStream stream, Bwidth bdw)
{
    XQueueHandle handle = GetXHandle(stream);
    auto hint = std::make_unique<xsched::sched::SetBandwidthHint>(
        (xsched::sched::Bwidth)bdw, handle);
    xsched::shim::XManager::GiveHint(std::move(hint));
}

namespace xsched::shim::vpi
{

VPIStatus XEventRecord(VPIEvent event, VPIStream stream)
{
    if (event == nullptr || stream == nullptr) {
        return Driver::EventRecord(event, stream);
    }

    auto xevent = std::make_shared<VpiEventRecordCommand>(event);
    VPIStatus ret = SubmitVpiCommand(xevent, stream);
    if (UNLIKELY(ret != VPI_SUCCESS)) return ret;
    
    xevent_mutex.lock();
    xevents[event] = xevent;
    xevent_mutex.unlock();

    return ret;
}

VPIStatus XEventSync(VPIEvent event)
{
    if (event == nullptr) {
        return Driver::EventSync(event);
    }

    xevent_mutex.lock();
    auto it = xevents.find(event);

    if (it == xevents.end()) {
        xevent_mutex.unlock();
        return Driver::EventSync(event);
    }

    auto xevent = it->second;
    xevent_mutex.unlock();

    xevent->Synchronize();
    return VPI_SUCCESS;
}

void XEventDestroy(VPIEvent event)
{
    if (event == nullptr) {
        Driver::EventDestroy(event);
        return;
    }

    xevent_mutex.lock();
    xevents.erase(event);
    xevent_mutex.unlock();

    Driver::EventDestroy(event);
}

VPIStatus XStreamSync(VPIStream stream)
{
    if (XManager::Synchronize(GetXHandle(stream))) return VPI_SUCCESS;
    return Driver::StreamSync(stream);
}

VPIStatus XStreamCreate(uint64_t flags, VPIStream *stream)
{
    // TODO: generate sched event
    return Driver::StreamCreate(flags, stream);
}

void XStreamDestroy(VPIStream stream)
{
    // TODO: generate sched event
    Driver::StreamDestroy(stream);
}

} // namespace xsched::shim::vpi
