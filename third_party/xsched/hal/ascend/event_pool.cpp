#include "hal/ascend/acl.h"
#include "hal/ascend/api.h"
#include "hal/ascend/event_pool.h"
#include "hal/ascend/acl_assert.h"

using namespace xsched::hal::ascend;

EventPool xsched::hal::ascend::g_event_pool; 

void *EventPool::Create()
{
    aclrtEvent event;
    ACL_ASSERT(Api::CreateEvent(&event));
    return event;
}
