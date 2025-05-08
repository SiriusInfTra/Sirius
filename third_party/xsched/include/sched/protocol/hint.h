#pragma once

#include <memory>
#include <cstddef>

#include "preempt/xqueue/xtype.h"

namespace xsched::sched
{

/// New HintTypes should be added here
/// when creating a new policy with new hints.
enum HintType
{
    kHintSetPriority    = 0,
    kHintSetBandwidth   = 1,
    kHintSetTimeslice   = 2,
};

struct HintMeta
{
    HintType type;
};

class Hint
{
public:
    Hint() = default;
    virtual ~Hint() = default;

    /// @brief Get the data of the Hint. MUST start with EventType.
    virtual const void *Data() const = 0;
    virtual size_t      Size() const = 0;
    virtual HintType    Type() const = 0;

    static std::unique_ptr<const Hint> CopyConstructor(const void *data);
};

} // namespace xsched::sched
