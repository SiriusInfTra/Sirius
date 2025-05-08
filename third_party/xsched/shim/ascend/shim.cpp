#include <mutex>
#include <unordered_map>

#include "utils/xassert.h"
#include "sched/policy/hpf.h"
#include "sched/policy/cbs.h"
#include "shim/ascend/shim.h"
#include "shim/ascend/xctrl.h"
#include "shim/common/agent.h"
#include "hal/ascend/acl_queue.h"
#include "preempt/xqueue/xqueue.h"

using namespace xsched::preempt;
using namespace xsched::hal::ascend;
using namespace xsched::shim::ascend;

static std::mutex xevent_mutex;
static std::unordered_map<aclrtEvent,
                          std::shared_ptr<AclEventRecordCommand>> xevents;

ACL_CTRL_FUNC uint64_t AclXQueueCreate(aclrtStream stream,
                                       int preempt_mode,
                                       int64_t queue_length,
                                       int64_t sync_interval)
{
    XASSERT(stream != nullptr, "cannot enable xpreempt on default stream");
    XASSERT(preempt_mode <= kPreemptModeStopSubmission,
            "only kPreemptModeStopSubmission is supported");
    
    XQueueHandle handle = GetXHandle(stream);
    XPreemptMode mode = (XPreemptMode)preempt_mode;
    auto acl_queue = std::make_shared<AclQueue>(mode, stream);
    auto xqueue = xsched::shim::XManager::CreateXQueue(acl_queue, handle,
        kDeviceAscend, mode, queue_length, sync_interval);
    
    return (uint64_t)handle;
}

ACL_CTRL_FUNC void AclXQueueDestroy(uint64_t handle)
{
    xsched::shim::XManager::DestroyXQueue(handle);
}

ACL_CTRL_FUNC void AclXQueueSuspend(aclrtStream stream, bool sync_hal_queue)
{
    XQueueHandle handle = GetXHandle(stream);
    xsched::shim::XManager::Suspend(handle, sync_hal_queue);
}

ACL_CTRL_FUNC void AclXQueueResume(aclrtStream stream, bool drop_commands)
{
    XQueueHandle handle = GetXHandle(stream);
    xsched::shim::XManager::Resume(handle, drop_commands);
}

ACL_CTRL_FUNC void AclXQueueSetPriority(aclrtStream stream, Prio prio)
{
    XQueueHandle handle = GetXHandle(stream);
    auto hint = std::make_unique<xsched::sched::SetPriorityHint>(
        (xsched::sched::Prio)prio, handle);
    xsched::shim::XManager::GiveHint(std::move(hint));
}

ACL_CTRL_FUNC void AclXQueueSetBandwidth(aclrtStream stream, Bwidth bdw)
{
    XQueueHandle handle = GetXHandle(stream);
    auto hint = std::make_unique<xsched::sched::SetBandwidthHint>(
        (xsched::sched::Bwidth)bdw, handle);
    xsched::shim::XManager::GiveHint(std::move(hint));
}

namespace xsched::shim::ascend
{

aclError XEventRecord(aclrtEvent event, aclrtStream stream)
{
    auto xevent = std::make_shared<AclEventRecordCommand>(event);
    aclError ret = SubmitAclCommand(xevent, stream);
    if (UNLIKELY(ret != ACL_SUCCESS)) return ret;
    
    xevent_mutex.lock();
    xevents[event] = xevent;
    xevent_mutex.unlock();

    return ret;
}

aclError XEventSynchronize(aclrtEvent event)
{
    xevent_mutex.lock();
    auto it = xevents.find(event);

    if (it == xevents.end()) {
        xevent_mutex.unlock();
        return Api::SynchronizeEvent(event);
    }

    auto xevent = it->second;
    xevent_mutex.unlock();

    xevent->Synchronize();
    return ACL_SUCCESS;
}

aclError XStreamSynchronize(aclrtStream stream)
{
    if (XManager::Synchronize(GetXHandle(stream))) return ACL_SUCCESS;
    return Api::SynchronizeStream(stream);
}

aclError XCreateStream(aclrtStream *stream)
{
    // TODO: generate sched event
    return Api::CreateStream(stream);
}

} // namespace xsched::shim::ascend
