#pragma once

#include <memory>
#include <functional>

#include "sched/policy/policy.h"
#include "sched/protocol/event.h"
#include "sched/protocol/operation.h"

namespace xsched::sched
{

typedef std::function<void(std::unique_ptr<const Operation>)> Executor;

enum SchedulerType
{
    kSchedulerLocal         = 0,
    kSchedulerGlobal        = 1,
    kSchedulerUserManaged   = 2,
};

class Scheduler
{
public:
    Scheduler(SchedulerType type): kType(type) {}
    virtual ~Scheduler() = default;

    virtual void Run() = 0;
    virtual void Stop() = 0;
    virtual void RecvEvent(std::unique_ptr<const Event> event) = 0;
    
    void SetExecutor(Executor executor) { executor_ = executor; }

    const SchedulerType kType;

protected:
    void Execute(std::unique_ptr<const Operation> operation);

private:
    Executor executor_ = nullptr;
};

std::unique_ptr<Scheduler> CreateScheduler();

} // namespace xsched::sched
