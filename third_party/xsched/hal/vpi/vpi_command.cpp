#include "utils/xassert.h"
#include "preempt/xqueue/xqueue.h"
#include "hal/vpi/event_pool.h"
#include "hal/vpi/vpi_assert.h"
#include "hal/vpi/vpi_command.h"

using namespace xsched::hal::vpi;

VpiCommand::~VpiCommand()
{
    if (following_event_ == nullptr) return;
    g_event_pool.Push(following_event_);
}

VPIStatus VpiCommand::EnqueueWrapper(VPIStream stream)
{
    VPIStatus ret = Enqueue(stream);
    if (UNLIKELY(ret != VPI_SUCCESS)) return ret;
    if (following_event_ != nullptr) { 
        ret = Driver::EventRecord(following_event_, stream);
    }
    return ret;
}

void VpiCommand::HalSynchronize()
{
    XASSERT(following_event_ != nullptr, "following_event_ is nullptr");
    VPI_ASSERT(Driver::EventSync(following_event_));
}

bool VpiCommand::HalSynchronizable()
{
    return following_event_ != nullptr;
}

bool VpiCommand::EnableHalSynchronization()
{
    following_event_ = (VPIEvent)g_event_pool.Pop();
    return following_event_ != nullptr;
}

VpiEventRecordCommand::VpiEventRecordCommand(VPIEvent event)
    : VpiCommand(preempt::kHalCommandTypeIdempotent), event_(event)
{
    XASSERT(event_ != nullptr, "vpi event should not be nullptr");
}


void VpiEventRecordCommand::HalSynchronize()
{
    VPI_ASSERT(Driver::EventSync(event_));
}

VPIStatus VpiEventRecordCommand::Enqueue(VPIStream stream)
{
    return Driver::EventRecord(event_, stream);
}
