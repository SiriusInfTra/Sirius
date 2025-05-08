#include "hal/vpi/event_pool.h"

using namespace xsched::hal::vpi;

EventPool xsched::hal::vpi::g_event_pool; 

void *EventPool::Create()
{
    VPIEvent event;
    VPI_ASSERT(Driver::EventCreate(VPI_EVENT_DISABLE_TIMESTAMP, &event));
    return event;
}
