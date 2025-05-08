#include "preempt/event/dispatcher.h"

using namespace xsched::preempt;

EventDispatcher xsched::preempt::g_event_dispatcher;

void EventDispatcher::Dispatch(std::unique_ptr<const sched::Event> event)
{
    for (auto &listener : listeners_) listener(std::move(event));
}

void EventDispatcher::AddListener(EventListener listener)
{
    listeners_.push_back(listener);
}
