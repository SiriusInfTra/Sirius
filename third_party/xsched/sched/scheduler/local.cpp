#include <cstring>
#include <algorithm>

#include "utils/xassert.h"
#include "sched/scheduler/local.h"

using namespace xsched::sched;
using namespace xsched::preempt;

LocalScheduler::LocalScheduler(PolicyType type)
    : Scheduler(kSchedulerLocal)
    , kPolicyType(type)
{
    policy_ = CreatePolicy(type);
    policy_->SetSuspendFunc(std::bind(&LocalScheduler::Suspend,
                                      this,
                                      std::placeholders::_1));
    policy_->SetResumeFunc(std::bind(&LocalScheduler::Resume,
                                     this,
                                     std::placeholders::_1));
    policy_->SetAddTimerFunc(std::bind(&LocalScheduler::AddTimer,
                                       this,
                                       std::placeholders::_1));
}

LocalScheduler::~LocalScheduler()
{
    Stop();
}

void LocalScheduler::Run()
{
    thread_ = std::make_unique<std::thread>(&LocalScheduler::Worker, this);
}

void LocalScheduler::Stop()
{
    if (thread_) {
        auto e = std::make_unique<TerminateEvent>();
        RecvEvent(std::move(e));
        thread_->join();
    }

    thread_ = nullptr;
    event_queue_.clear();
    timers_.clear();
}

void LocalScheduler::RecvEvent(std::unique_ptr<const Event> event)
{
    event_mtx_.lock();
    event_queue_.emplace_back(std::move(event));
    event_mtx_.unlock();
    event_cv_.notify_all();
}

void LocalScheduler::Worker()
{
    std::list<std::unique_ptr<const Event>> tmp_queue;
    std::unique_lock<std::mutex> lock(event_mtx_);

    while (true) {
        // wait for event or the first timer
        while (event_queue_.empty()) {
            if (timers_.empty()) {
                event_cv_.wait(lock);
                continue;
            }

            auto first_timer = timers_.front();
            auto now = std::chrono::system_clock::now();
            if (now < first_timer) {
                event_cv_.wait_until(lock, first_timer);
                continue;
            }

            while (!timers_.empty()) {
                if (now < timers_.front()) break;
                timers_.pop_front();
            }
            break;
        }
        
        // move events to tmp_queue
        while (!event_queue_.empty()) {
            auto event = std::move(event_queue_.front());
            event_queue_.pop_front();
            tmp_queue.push_back(std::move(event));
        }
        lock.unlock();

        // process events in tmp_queue
        while (!tmp_queue.empty()) {
            auto event = std::move(tmp_queue.front());
            tmp_queue.pop_front();

            if (UNLIKELY(event->Type() == kEventTerminate)) {
                tmp_queue.clear();
                return;
            }
            this->UpdateStatus(std::move(event));
        }

        policy_->Sched(status_);    // reschedule
        this->ExecuteOperations();  // find changes and execute
        std::sort(timers_.begin(), timers_.end());

        lock.lock();
    }
}

void LocalScheduler::ExecuteOperations()
{
    for (auto &status : status_.process_status) {
        if (status.second->running_xqueues.empty() &&
            status.second->suspended_xqueues.empty()) continue;
        auto op = std::make_unique<SchedOperation>(*status.second);
        Execute(std::move(op));
    }
}

void LocalScheduler::CreateXQueueStatus(preempt::XQueueHandle handle,
                                        preempt::XDevice device,
                                        PID pid, bool ready)
{
    auto status = std::make_unique<XQueueStatus>();
    status->handle = handle;
    status->device = device;
    status->pid = pid;
    status->ready = ready;
    status->suspended = false;
    status_.xqueue_status[handle] = std::move(status);

    // if process status not exist, create one
    auto it = status_.process_status.find(pid);
    if (it == status_.process_status.end()) {
        auto process_status = std::make_unique<ProcessStatus>();
        process_status->pid = pid;
        status_.process_status[pid] = std::move(process_status);
        it = status_.process_status.find(pid);
    }
    it->second->running_xqueues.insert(handle);
}

void LocalScheduler::UpdateStatus(std::unique_ptr<const Event> event)
{
    switch (event->Type()) {
    case kEventHint:
    {
        auto e = (const HintEvent *)event.get();
        policy_->GiveHint(std::move(e->GetHint()));
        break;
    }
    case kEventProcessDestroy:
    {
        auto e = (const ProcessDestroyEvent *)event.get();
        PID pid = e->Pid();
        auto pit = status_.process_status.find(pid);
        if (pit == status_.process_status.end()) break;

        for (auto &handle : pit->second->running_xqueues) {
            status_.xqueue_status.erase(handle);
        }
        for (auto &handle : pit->second->suspended_xqueues) {
            status_.xqueue_status.erase(handle);
        }
        status_.process_status.erase(pit);
        break;
    }
    case kEventXQueueCreate:
    {
        auto e = (const XQueueCreateEvent *)event.get();
        XINFO("XQueue (0x%lx) from process (%u) created",
              e->Handle(), e->Pid());
        auto it = status_.xqueue_status.find(e->Handle());
        if (it == status_.xqueue_status.end()) {
            // if xqueue status not exist, create one
            CreateXQueueStatus(e->Handle(), e->Device(), e->Pid(), false);
        } else {
            it->second->device = e->Device();
        }
        break;
    }
    case kEventXQueueDestroy:
    {
        auto e = (const XQueueDestroyEvent *)event.get();
        XINFO("XQueue (0x%lx) from process (%u) destoryed",
              e->Handle(), e->Pid());
        XQueueHandle handle = e->Handle();
        auto qit = status_.xqueue_status.find(handle);
        if (qit == status_.xqueue_status.end()) break;

        PID pid = qit->second->pid;
        XASSERT(e->Pid() == pid, "pid not match");
        auto pit = status_.process_status.find(pid);
        if (pit == status_.process_status.end()) break;

        pit->second->running_xqueues.erase(handle);
        pit->second->suspended_xqueues.erase(handle);
        status_.xqueue_status.erase(qit);
        break;
    }
    case kEventXQueueReady:
    {
        auto e = (const XQueueReadyEvent *)event.get();
        auto it = status_.xqueue_status.find(e->Handle());
        if (it == status_.xqueue_status.end()) {
            // if xqueue status not exist, create one
            CreateXQueueStatus(e->Handle(), kDeviceUnknown, e->Pid(), true);
        } else {
            it->second->ready = true;
        }
        break;
    }
    case kEventXQueueIdle:
    {
        auto e = (const XQueueIdleEvent *)event.get();
        auto it = status_.xqueue_status.find(e->Handle());
        if (it == status_.xqueue_status.end()) {
            // if xqueue status not exist, create one
            CreateXQueueStatus(e->Handle(), kDeviceUnknown, e->Pid(), false);
        } else {
            it->second->ready = false;
        }
        break;
    }
    case kEventStatusQuery:
    {
        auto e = (const StatusQueryEvent *)event.get();
        StatusQuery *query = e->QueryData();
        for (auto &status : status_.xqueue_status) {
            query->status_.emplace_back(
                std::make_unique<XQueueStatus>(*status.second));
        }
        query->Notify();
        break;
    }
    default:
        break;
    }
}

void LocalScheduler::Suspend(XQueueHandle handle)
{
    auto qit = status_.xqueue_status.find(handle);
    if (qit == status_.xqueue_status.end()) return;

    qit->second->suspended = true;
    PID pid = qit->second->pid;

    auto pit = status_.process_status.find(pid);
    if (pit == status_.process_status.end()) return;
    
    pit->second->running_xqueues.erase(handle);
    pit->second->suspended_xqueues.insert(handle);
}

void LocalScheduler::Resume(XQueueHandle handle)
{
    auto qit = status_.xqueue_status.find(handle);
    if (qit == status_.xqueue_status.end()) return;

    qit->second->suspended = false;
    PID pid = qit->second->pid;

    auto pit = status_.process_status.find(pid);
    if (pit == status_.process_status.end()) return;
    
    pit->second->running_xqueues.insert(handle);
    pit->second->suspended_xqueues.erase(handle);
}

void LocalScheduler::AddTimer(
    const std::chrono::system_clock::time_point time_point)
{
    timers_.push_back(time_point);
}
