#pragma once

#include <map>
#include <chrono>
#include <cstdint>

#include "sched/protocol/hint.h"
#include "sched/policy/policy.h"
#include "preempt/xqueue/xtype.h"

namespace xsched::sched
{

typedef int32_t Bwidth;

#define BANDWIDTH_MIN 0
#define BANDWIDTH_MAX 32

class SetBandwidthHint : public Hint
{
public:
    SetBandwidthHint(const void *data)
        : data_(*(const HintData *)data) {}
    SetBandwidthHint(Bwidth bdw, preempt::XQueueHandle handle)
        : data_{
            .meta { .type = kHintSetBandwidth },
            .bandwidth = bdw,
            .handle = handle
        } {}
    virtual ~SetBandwidthHint() = default;

    virtual const void *Data() const override { return &data_; }
    virtual size_t      Size() const override { return sizeof(data_); }
    virtual HintType    Type() const override { return kHintSetBandwidth; }

    Bwidth Bandwidth() const { return data_.bandwidth; }
    preempt::XQueueHandle Handle() const { return data_.handle; }

private:
    struct HintData
    {
        HintMeta meta;
        Bwidth   bandwidth;
        preempt::XQueueHandle handle;
    };

    HintData data_;
};

class SetTimesliceHint : public Hint
{
public:
    SetTimesliceHint(const void *data)
        : data_(*(const HintData *)data) {}
    SetTimesliceHint(uint64_t timeslice_us)
        : data_{
            .meta { .type = kHintSetTimeslice },
            .timeslice_us = timeslice_us
        } {}
    virtual ~SetTimesliceHint() = default;

    virtual const void *Data() const override { return &data_; }
    virtual size_t      Size() const override { return sizeof(data_); }
    virtual HintType    Type() const override { return kHintSetTimeslice; }

    uint64_t Timeslice() const { return data_.timeslice_us; }

private:
    struct HintData
    {
        HintMeta meta;
        uint64_t timeslice_us;
    };

    HintData data_;
};

class ConstantBandwidthServer : public Policy
{
public:
    ConstantBandwidthServer() : Policy(kPolicyConstantBandwidthServer) {}
    virtual ~ConstantBandwidthServer() = default;

    virtual void Sched(const Status &status) override;
    virtual void GiveHint(std::unique_ptr<const Hint> hint) override;

private:
    std::chrono::microseconds GetBudget(Bwidth bdw);
    void SwitchToAny(const Status &status);
    void SwitchTo(preempt::XQueueHandle handle,
                  Bwidth bdw, const Status &status);

    preempt::XQueueHandle cur_running_ = 0;
    std::chrono::system_clock::time_point cur_end_;
    std::map<preempt::XQueueHandle, Bwidth> bandwidths_;
    std::chrono::microseconds timeslice_ = std::chrono::microseconds(0x2000);
};

} // namespace xsched::sched
