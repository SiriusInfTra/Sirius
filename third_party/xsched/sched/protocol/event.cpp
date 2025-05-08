#include <cstring>

#include "utils/xassert.h"
#include "sched/protocol/event.h"

using namespace xsched::sched;

std::unique_ptr<const Event> Event::CopyConstructor(const void *data)
{
    auto meta = (const EventMeta *)data;
    switch (meta->type)
    {
    case kEventHint:
        return std::make_unique<HintEvent>(data);
    case kEventTerminate:
        return std::make_unique<TerminateEvent>(data);
    case kEventProcessCreate:
        return std::make_unique<ProcessCreateEvent>(data);
    case kEventProcessDestroy:
        return std::make_unique<ProcessDestroyEvent>(data);
    case kEventXQueueCreate:
        return std::make_unique<XQueueCreateEvent>(data);
    case kEventXQueueDestroy:
        return std::make_unique<XQueueDestroyEvent>(data);
    case kEventXQueueReady:
        return std::make_unique<XQueueReadyEvent>(data);
    case kEventXQueueIdle:
        return std::make_unique<XQueueIdleEvent>(data);
    case kEventStatusQuery:
        return std::make_unique<StatusQueryEvent>(data);
    default:
        XASSERT(false, "unknown event type: %d", meta->type);
        return nullptr;
    }
}

HintEvent::HintEvent(const void *data)
{
    size_t size = ((const EventData *)data)->size;
    EventData *new_data = (EventData *)malloc(size);
    memcpy(new_data, data, size);
    data_ = new_data;
}

HintEvent::HintEvent(std::unique_ptr<const Hint> hint)
{
    size_t size = offsetof(EventData, hint_data) + hint->Size();
    EventData *data = (EventData *)malloc(size);

    data->meta.type = kEventHint;
    data->meta.pid = GetProcessId();
    data->size = size;
    memcpy(data->hint_data, hint->Data(), hint->Size());

    data_ = data;
}

HintEvent::~HintEvent()
{
    if (data_) free((void *)data_);
}

StatusQuery *StatusQueryEvent::QueryData() const
{
    XASSERT(data_.query_data != nullptr,
            "query data should not be nullptr");
    XASSERT(data_.meta.pid == GetProcessId(),
            "query data should only be accessed by the process creating it");
    return data_.query_data;
}
