#pragma once

#include "sched/scheduler/scheduler.h"

namespace xsched::sched
{

class UserManagedScheduler : public Scheduler
{
public:
    UserManagedScheduler(): Scheduler(kSchedulerUserManaged) {}
    virtual ~UserManagedScheduler() = default;

    virtual void Run() override {}
    virtual void Stop() override {}
    virtual void RecvEvent(std::unique_ptr<const Event>) override {}
};

} // namespace xsched::sched
