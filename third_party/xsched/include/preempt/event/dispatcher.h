#pragma once

#include <vector>
#include <functional>

#include "sched/protocol/event.h"

namespace xsched::preempt
{

typedef std::function<void(std::unique_ptr<const sched::Event>)> EventListener;

class EventDispatcher
{
public:
    EventDispatcher() = default;
    ~EventDispatcher() = default;

    void Dispatch(std::unique_ptr<const sched::Event> event);
    void AddListener(EventListener listener);

private:
    std::vector<EventListener> listeners_;
};

extern EventDispatcher g_event_dispatcher;

} // namespace xsched::preempt
