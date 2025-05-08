#pragma once

#include <chrono>
#include <memory>
#include <functional>

#include "preempt/xqueue/xtype.h"
#include "sched/protocol/hint.h"
#include "sched/protocol/status.h"

namespace xsched::sched
{

typedef std::chrono::system_clock::time_point TimePoint;
typedef std::function<void (const TimePoint)> AddTimerFunc;
typedef std::function<void (const preempt::XQueueHandle)> OperateFunc;

/// A new PolicyType should be added here
/// when creating a new policy.
enum PolicyType
{
    kPolicyGlobal                  = 0,
    kPolicyUserManaged             = 1,
    kPolicyInternalMax             = 2,

    kPolicyHighestPriorityFirst    = 3,
    kPolicyConstantBandwidthServer = 4,
};

class Policy
{
public:
    Policy(PolicyType type): kType(type) {}
    virtual ~Policy() = default;

    void SetSuspendFunc(OperateFunc suspend) { suspend_func_ = suspend; }
    void SetResumeFunc(OperateFunc resume) { resume_func_ = resume; }
    void SetAddTimerFunc(AddTimerFunc add_timer) { add_timer_func_ = add_timer; }

    virtual void Sched(const Status &status) = 0;
    virtual void GiveHint(std::unique_ptr<const Hint> hint) = 0;

    const PolicyType kType;

protected:
    void Suspend(preempt::XQueueHandle xqueue);
    void Resume(preempt::XQueueHandle xqueue);
    void AddTimer(const TimePoint time_point);

private:
    OperateFunc suspend_func_ = nullptr;
    OperateFunc resume_func_ = nullptr;
    AddTimerFunc add_timer_func_ = nullptr;
};

std::unique_ptr<Policy> CreatePolicy(PolicyType type);

} // namespace xsched::sched
