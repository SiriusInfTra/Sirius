#pragma once

#include <mutex>
#include <memory>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <condition_variable>

#include "utils/common.h"
#include "preempt/xqueue/xtype.h"

namespace xsched::sched
{

struct XQueueStatus
{
    preempt::XQueueHandle handle;
    preempt::XDevice device;
    PID pid;
    bool ready;
    bool suspended;
};

struct ProcessStatus
{
    PID pid;
    std::unordered_set<preempt::XQueueHandle> running_xqueues;
    std::unordered_set<preempt::XQueueHandle> suspended_xqueues;
};

struct Status
{
    std::unordered_map
        <preempt::XQueueHandle, std::unique_ptr<XQueueStatus>> xqueue_status;
    std::unordered_map
        <PID, std::unique_ptr<ProcessStatus>> process_status;
};

class StatusQuery
{
public:
    StatusQuery() = default;
    ~StatusQuery() = default;

    void Wait();
    void Notify();
    void Reset();

    std::vector<std::unique_ptr<XQueueStatus>> status_;

private:
    bool ready_ = false;
    std::mutex mtx_;
    std::condition_variable cv_;
};

} // namespace xsched::sched
