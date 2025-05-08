#include <cstdlib>

#include "utils/log.h"
#include "utils/xassert.h"
#include "sched/scheduler/local.h"
#include "sched/scheduler/global.h"
#include "sched/scheduler/scheduler.h"
#include "sched/scheduler/user_managed.h"
#include "sched/protocol/protocol.h"

using namespace xsched::sched;

void Scheduler::Execute(std::unique_ptr<const Operation> operation)
{
    if (executor_) return executor_(std::move(operation));
    XDEBG("executor not set");
}

std::unique_ptr<Scheduler> xsched::sched::CreateScheduler()
{
    char *policy_name = std::getenv(kPolicyEnvVarName.c_str());
    if (policy_name == nullptr) {
        return std::make_unique<UserManagedScheduler>();
    }

    auto it = kPolicyTypeMap.find(policy_name);
    if (it == kPolicyTypeMap.end() || it->second == kPolicyUserManaged) {
        return std::make_unique<UserManagedScheduler>();
    }

    if (it->second == kPolicyGlobal) {
        return std::make_unique<GlobalScheduler>();
    }

    XASSERT(it->second > kPolicyInternalMax,
            "must be a customized policy");
    return std::make_unique<LocalScheduler>(it->second);
}
