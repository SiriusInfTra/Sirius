#include <cstring>

#include "utils/xassert.h"
#include "sched/protocol/hint.h"
#include "sched/policy/hpf.h"
#include "sched/policy/cbs.h"

using namespace xsched::sched;

std::unique_ptr<const Hint> Hint::CopyConstructor(const void *data)
{
    auto meta = (const HintMeta *)data;
    switch (meta->type)
    {
    /// New cases handling new HintTypes should be added here
    /// when creating a new policy with new hints.
    case kHintSetPriority:
        return std::make_unique<SetPriorityHint>(data);
    case kHintSetBandwidth:
        return std::make_unique<SetBandwidthHint>(data);
    case kHintSetTimeslice:
        return std::make_unique<SetTimesliceHint>(data);
    default:
        XASSERT(false, "unknown hint type: %d", meta->type);
        return nullptr;
    }
}
