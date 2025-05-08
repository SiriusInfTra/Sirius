#include "hal/vpi/driver.h"
#include "hal/vpi/vpi_queue.h"
#include "hal/vpi/vpi_assert.h"
#include "hal/vpi/vpi_command.h"

using namespace xsched::hal::vpi;

VpiQueue::VpiQueue(preempt::XPreemptMode mode, VPIStream stream)
    : mode_(mode), stream_(stream)
{
    // make sure no tasks are running on stream_
    VPI_ASSERT(Driver::StreamSync(stream_));
}

void VpiQueue::HalSynchronize()
{
    VPI_ASSERT(Driver::StreamSync(stream_));
}

void VpiQueue::HalSubmit(std::shared_ptr<preempt::HalCommand> hal_command)
{
    auto vpi_command = std::dynamic_pointer_cast<VpiCommand>(hal_command);
    XASSERT(vpi_command != nullptr, "hal_command is not a VpiCommand");
    VPI_ASSERT(vpi_command->EnqueueWrapper(stream_));
}
