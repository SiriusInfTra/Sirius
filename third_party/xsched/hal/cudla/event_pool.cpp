#include "hal/cudla/event_pool.h"

using namespace xsched::hal::cudla;

EventPool xsched::hal::cudla::g_event_pool; 

void *EventPool::Create()
{
    cudaEvent_t event;
    CUDART_ASSERT(RtDriver::EventCreateWithFlags(&event,
        cudaEventBlockingSync | cudaEventDisableTiming));
    return event;
}
