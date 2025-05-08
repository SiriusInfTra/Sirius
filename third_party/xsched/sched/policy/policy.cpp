#include <cstdlib>

#include "utils/log.h"
#include "utils/xassert.h"
#include "sched/protocol/protocol.h"
#include "sched/policy/policy.h"
#include "sched/policy/hpf.h"
#include "sched/policy/cbs.h"

using namespace xsched::sched;
using namespace xsched::preempt;

void Policy::Suspend(XQueueHandle xqueue)
{
    if (suspend_func_) return suspend_func_(xqueue);
    XDEBG("suspend function not set");
}

void Policy::Resume(XQueueHandle xqueue)
{
    if (resume_func_) return resume_func_(xqueue);
    XDEBG("resume function not set");
}

void Policy::AddTimer(const TimePoint time_point)
{
    if (add_timer_func_) return add_timer_func_(time_point);
    XDEBG("add timer function not set");
}

std::unique_ptr<Policy> xsched::sched::CreatePolicy(PolicyType type)
{
    /// A new case handling new PolicyType should be added here
    /// when creating a new policy.
    switch (type) {
        case kPolicyHighestPriorityFirst:
            return std::make_unique<HighestPriorityFirstPolicy>();
        case kPolicyConstantBandwidthServer:
            return std::make_unique<ConstantBandwidthServer>();
        default:
            XASSERT(false, "invalid policy type: %d", type);
            return nullptr;
    }
}
