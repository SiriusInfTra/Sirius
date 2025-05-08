#pragma once

#include "hal/vpi/vpi.h"
#include "preempt/xqueue/xtype.h"
#include "preempt/hal/hal_queue.h"

namespace xsched::hal::vpi
{

class VpiQueue : public preempt::HalQueue
{
public:
    VpiQueue(preempt::XPreemptMode mode, VPIStream stream);
    virtual ~VpiQueue() = default;

    VPIStream GetCudlaStream() const { return stream_; }
    
    virtual void HalSynchronize() override;
    virtual void HalSubmit(
        std::shared_ptr<preempt::HalCommand> hal_command) override;

private:
    const preempt::XPreemptMode mode_;
    const VPIStream stream_;
};

} // namespace xsched::hal::vpi
