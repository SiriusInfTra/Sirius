#include "utils/xassert.h"
#include "sched/policy/cbs.h"

using namespace xsched::sched;
using namespace xsched::preempt;

void ConstantBandwidthServer::Sched(const Status &status)
{
    // delete all destoryed xqueues and their bandwidths
    for (auto it = bandwidths_.begin(); it != bandwidths_.end();) {
        XQueueHandle xqueue = it->first;
        if (status.xqueue_status.find(xqueue) == status.xqueue_status.end()) {
            it = bandwidths_.erase(it);
        } else {
            ++it;
        }
    }

    // delete destoryed current running xqueue
    if (cur_running_ != 0 &&
        status.xqueue_status.find(cur_running_)
            == status.xqueue_status.end()) {
        cur_running_ = 0;
    }

    if (bandwidths_.empty()) return;

    if (cur_running_ == 0) {
        // nothing is running
        SwitchToAny(status);
        return;
    }

    auto xit = status.xqueue_status.find(cur_running_);
    if (xit == status.xqueue_status.end()) {
        // current xqueue is not found
        auto bit = bandwidths_.find(cur_running_);
        XASSERT(bit != bandwidths_.end(),
                "bandwidth of XQueue 0x%lx not found.", cur_running_);
        bandwidths_.erase(bit);
        SwitchToAny(status);
        return;
    }

    auto now = std::chrono::system_clock::now();
    if (now < cur_end_ && xit->second->ready) return;

    auto bit = bandwidths_.find(cur_running_);
    XASSERT(bit != bandwidths_.end(),
            "bandwidth of XQueue 0x%lx not found.", cur_running_);
    
    // current xqueue has finished its time slice
    // select the next xqueue to run
    for (++bit; bit != bandwidths_.end();) {
        auto xit = status.xqueue_status.find(bit->first);
        if (xit == status.xqueue_status.end()) {
            // xqueue not found
            bandwidths_.erase(bit++);
            continue;
        }
        if (!xit->second->ready) {
            // not running check the next one
            ++bit;
            continue;
        }
        SwitchTo(bit->first, bit->second, status);
        return;
    }

    for (bit = bandwidths_.begin(); bit != bandwidths_.end();) {
        if (bit->first == cur_running_) break;
        
        auto xit = status.xqueue_status.find(bit->first);
        if (xit == status.xqueue_status.end()) {
            // xqueue not found
            bandwidths_.erase(bit++);
            continue;
        }
        if (!xit->second->ready) {
            // not running check the next one
            ++bit;
            continue;
        }
        SwitchTo(bit->first, bit->second, status);
        return;
    }

    // checked a round, no xqueue to run
    cur_running_ = 0;
}

void ConstantBandwidthServer::GiveHint(std::unique_ptr<const Hint> hint)
{
    switch (hint->Type())
    {
    case kHintSetBandwidth:
    {
        const SetBandwidthHint *h = (const SetBandwidthHint *)hint.get();
        Bwidth bdw = h->Bandwidth();
        if (bdw < BANDWIDTH_MIN || bdw > BANDWIDTH_MAX) {
            XWARN("invalid bandwidth %d", bdw);
            break;
        }
        bandwidths_[h->Handle()] = h->Bandwidth();
        break;
    }
    case kHintSetTimeslice:
    {
        const SetTimesliceHint *h = (const SetTimesliceHint *)hint.get();
        timeslice_ = std::chrono::microseconds(h->Timeslice());
        break;
    }
    default:
        XWARN("unsupported hint type: %d", hint->Type());
        break;
    }
}

std::chrono::microseconds ConstantBandwidthServer::GetBudget(Bwidth bdw)
{
    Bwidth totalBdw = 0;
    int64_t totalUs = timeslice_.count();
    for (const auto &xqueue : bandwidths_) { totalBdw += xqueue.second; }
    return std::chrono::microseconds(totalUs * bdw / totalBdw);
}

void ConstantBandwidthServer::SwitchToAny(const Status &status)
{
    cur_running_ = 0;
    bool selected = false;
    for (const auto &xqueue : status.xqueue_status) {
        if (selected || !xqueue.second->ready) {
            this->Suspend(xqueue.first);
            continue;
        }

        auto it = bandwidths_.find(xqueue.first);
        if (it == bandwidths_.end()) {
            this->Suspend(xqueue.first);
            continue;
        }

        selected = true;
        this->Resume(xqueue.first);
        cur_running_ = xqueue.first;
        cur_end_ = std::chrono::system_clock::now() + GetBudget(it->second);
        this->AddTimer(cur_end_);
    }
}

void ConstantBandwidthServer::SwitchTo(preempt::XQueueHandle handle,
                                       Bwidth bdw,
                                       const Status &status)
{
    for (const auto &xqueue : status.xqueue_status) {
        if (xqueue.first == handle) continue;
        this->Suspend(xqueue.first);
    }

    this->Resume(handle);
    cur_running_ = handle;
    cur_end_ = std::chrono::system_clock::now() + GetBudget(bdw);
    this->AddTimer(cur_end_);
}
