#include "hal/cudla/driver.h"
#include "hal/cudla/cudla_queue.h"
#include "hal/cudla/cudla_assert.h"
#include "hal/cudla/cudla_command.h"

using namespace xsched::hal::cudla;

CudlaQueue::CudlaQueue(preempt::XPreemptMode mode, cudaStream_t stream)
    : mode_(mode), stream_(stream)
{
    // make sure no tasks are running on stream_
    CUDART_ASSERT(RtDriver::StreamSynchronize(stream_));
}

void CudlaQueue::HalSynchronize()
{
    CUDART_ASSERT(RtDriver::StreamSynchronize(stream_));
}

void CudlaQueue::HalSubmit(std::shared_ptr<preempt::HalCommand> hal_command)
{
    auto cudla_command = std::dynamic_pointer_cast<CudlaCommand>(hal_command);
    XASSERT(cudla_command != nullptr, "hal_command is not a CudlaCommand");
    CUDART_ASSERT(cudla_command->EnqueueWrapper(stream_));
}
