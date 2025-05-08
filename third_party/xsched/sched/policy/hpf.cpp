#include <map>

#include "utils/xassert.h"
#include "sched/policy/hpf.h"

using namespace xsched::sched;
using namespace xsched::preempt;

void HighestPriorityFirstPolicy::Sched(const Status &status)
{
    // find the highest priority task of each device
    std::map<XDevice, Prio> running_prio_max;
    for (auto &status : status.xqueue_status) {
        XQueueHandle handle = status.second->handle;
        Prio priority = PRIO_MIN;
        auto it = priorities_.find(handle);
        // if priority not found, use minimal priority
        if (it == priorities_.end()) priorities_[handle] = priority;
        else priority = it->second;

        if (!status.second->ready) continue;
        auto prio_it = running_prio_max.find(status.second->device);
        if (prio_it == running_prio_max.end()) {
            running_prio_max[status.second->device] = priority;
        } else if (priority > prio_it->second) {
            prio_it->second = priority;
        }
    }

    // suspend all xqueues with lower priority
    // and resume all xqueues with higher priority
    for (auto &status : status.xqueue_status) {
        XQueueHandle handle = status.second->handle;
        auto it = priorities_.find(handle);
        XASSERT(it != priorities_.end(),
                "priority of XQueue 0x%lx not found.", handle);
        Prio priority = it->second;

        Prio prio_max = PRIO_MIN;
        auto prio_it = running_prio_max.find(status.second->device);
        if (prio_it != running_prio_max.end()) {
            prio_max = prio_it->second;
        }

        if (priority < prio_max) {
            this->Suspend(handle);
        } else {
            this->Resume(handle);
        }
    }
}

void HighestPriorityFirstPolicy::GiveHint(std::unique_ptr<const Hint> hint)
{
    if (hint->Type() != kHintSetPriority) return;
    const SetPriorityHint *h = (const SetPriorityHint *)hint.get();

    Prio priority = h->Priority();
    if (priority < PRIO_MIN) priority = PRIO_MIN;
    if (priority > PRIO_MAX) priority = PRIO_MAX;
    if (priority != h->Priority()) {
        XWARN("priority %d for XQueue 0x%lu is invalid, "
              "valid range: [%d, %d], priority overide to %d",
              h->Priority(), h->Handle(), PRIO_MIN, PRIO_MAX, priority);
    }

    priorities_[h->Handle()] = priority;
}
