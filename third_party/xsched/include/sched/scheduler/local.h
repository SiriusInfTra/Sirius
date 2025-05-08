#pragma once

#include <list>
#include <queue>
#include <mutex>
#include <thread>
#include <chrono>
#include <memory>
#include <condition_variable>

#include "utils/common.h"
#include "sched/policy/policy.h"
#include "sched/protocol/status.h"
#include "sched/scheduler/scheduler.h"

namespace xsched::sched
{

class LocalScheduler : public Scheduler
{
public:
    LocalScheduler(PolicyType type);
    virtual ~LocalScheduler();

    virtual void Run() override;
    virtual void Stop() override;
    virtual void RecvEvent(std::unique_ptr<const Event> event) override;

    const PolicyType kPolicyType;

private:
    void Worker();
    void ExecuteOperations();
    void CreateXQueueStatus(preempt::XQueueHandle handle,
                            preempt::XDevice device,
                            PID pid, bool ready);
    void UpdateStatus(std::unique_ptr<const Event> event);
    void Suspend(preempt::XQueueHandle handle);
    void Resume(preempt::XQueueHandle handle);
    void AddTimer(const std::chrono::system_clock::time_point time_point);

    std::unique_ptr<Policy> policy_ = nullptr;
    std::unique_ptr<std::thread> thread_ = nullptr;

    Status status_;
    std::mutex event_mtx_;
    std::condition_variable event_cv_;
    std::list<std::unique_ptr<const Event>> event_queue_;
    std::deque<std::chrono::system_clock::time_point> timers_;
};

} // namespace xsched::sched
