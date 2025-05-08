#pragma once

#include <cstdint>
#include <unordered_map>

#include "sched/protocol/hint.h"
#include "sched/policy/policy.h"
#include "preempt/xqueue/xtype.h"

namespace xsched::sched
{

typedef int32_t Prio;

#define PRIO_MIN -16
#define PRIO_MAX  16

class SetPriorityHint : public Hint
{
public:
    SetPriorityHint(const void *data)
        : data_(*(const HintData *)data) {}
    SetPriorityHint(Prio priority, preempt::XQueueHandle handle)
        : data_{
            .meta { .type = kHintSetPriority },
            .priority = priority,
            .handle = handle
        } {}
    virtual ~SetPriorityHint() = default;

    virtual const void *Data() const override { return &data_; }
    virtual size_t      Size() const override { return sizeof(data_); }
    virtual HintType    Type() const override { return kHintSetPriority; }

    Prio Priority() const { return data_.priority; }
    preempt::XQueueHandle Handle() const { return data_.handle; }

private:
    struct HintData
    {
        HintMeta meta;
        Prio     priority;
        preempt::XQueueHandle handle;
    };

    HintData data_;
};

class HighestPriorityFirstPolicy : public Policy
{
public:
    HighestPriorityFirstPolicy(): Policy(kPolicyHighestPriorityFirst) {}
    virtual ~HighestPriorityFirstPolicy() = default;

    virtual void Sched(const Status &status) override;
    virtual void GiveHint(std::unique_ptr<const Hint> hint) override;

private:
    std::unordered_map<preempt::XQueueHandle, Prio> priorities_;
};

} // namespace xsched::sched
