#pragma once

#include "hal/cudla/cudla.h"
#include "hal/cudla/cudart.h"
#include "preempt/xqueue/xtype.h"
#include "preempt/hal/hal_queue.h"

namespace xsched::hal::cudla
{

class CudlaQueue : public preempt::HalQueue
{
public:
    CudlaQueue(preempt::XPreemptMode mode, cudaStream_t stream);
    virtual ~CudlaQueue() = default;

    cudaStream_t GetCudlaStream() const { return stream_; }
    
    virtual void HalSynchronize() override;
    virtual void HalSubmit(
        std::shared_ptr<preempt::HalCommand> hal_command) override;

private:
    const preempt::XPreemptMode mode_;
    const cudaStream_t stream_;
};

} // namespace xsched::hal::cudla
