#pragma once

#include "hal/cuda/cuda.h"
#include "hal/cuda/preempt.h"
#include "hal/cuda/cuda_command.h"
#include "preempt/xqueue/xtype.h"
#include "preempt/hal/hal_queue.h"

namespace xsched::hal::cuda
{

class CudaQueue : public preempt::HalQueue
{
public:
    CudaQueue(preempt::XPreemptMode mode);
    CudaQueue(preempt::XPreemptMode mode, CUstream stream);
    virtual ~CudaQueue() = default;

    CUstream GetCudaStream() const;

    virtual void OnInitialize() override;
    virtual void OnSubmit(
        std::shared_ptr<preempt::HalCommand> hal_command) override;

    virtual void HalSynchronize() override;
    virtual void HalSubmit(
        std::shared_ptr<preempt::HalCommand> hal_command) override;

    virtual void Deactivate() override;
    virtual void Reactivate(const preempt::CommandLog &log) override;
    virtual void Interrupt() override;

    static CUresult LaunchKernelNormal(
        std::shared_ptr<CudaKernelLaunchCommand> kernel, CUstream stream);

private:
    const preempt::XPreemptMode mode_;
    
    CUstream stream_ = nullptr;
    CUcontext context_ = nullptr;
    std::unique_ptr<PreemptManager> preempt_manager_ = nullptr;
};

} // namespace xsched::hal::cuda
