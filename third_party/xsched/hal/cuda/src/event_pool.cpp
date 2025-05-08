#include "hal/cuda/event_pool.h"

using namespace xsched::hal::cuda;

EventPool xsched::hal::cuda::g_event_pool; 

void *EventPool::Create()
{
    CUevent event;
    CUDA_ASSERT(Driver::EventCreate(&event,
        CU_EVENT_BLOCKING_SYNC | CU_EVENT_DISABLE_TIMING));
    return event;
}
