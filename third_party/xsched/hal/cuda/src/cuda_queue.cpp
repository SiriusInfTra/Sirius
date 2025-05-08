#include "utils/xassert.h"
#include "hal/cuda/driver.h"
#include "hal/cuda/cuda_queue.h"
#include "hal/cuda/cuda_assert.h"
#include "hal/cuda/cuda_command.h"

using namespace xsched::preempt;
using namespace xsched::hal::cuda;

CudaQueue::CudaQueue(XPreemptMode mode)
    : mode_(mode)
{
    CUDA_ASSERT(Driver::StreamCreate(&stream_, 0));
    CUDA_ASSERT(Driver::StreamGetCtx(stream_, &context_));
    preempt_manager_
        = std::make_unique<PreemptManager>(mode_, context_, stream_);
}

CudaQueue::CudaQueue(XPreemptMode mode, CUstream stream)
    : mode_(mode)
    , stream_(stream)
{
    // make sure no commands are running on stream_
    CUDA_ASSERT(Driver::StreamSynchronize(stream_));
    CUDA_ASSERT(Driver::StreamGetCtx(stream_, &context_));
    preempt_manager_
        = std::make_unique<PreemptManager>(mode_, context_, stream_);
}

CUstream CudaQueue::GetCudaStream() const
{
    return stream_;
}

void CudaQueue::OnInitialize()
{
    CUDA_ASSERT(Driver::CtxSetCurrent(context_));
}

void CudaQueue::OnSubmit(std::shared_ptr<HalCommand> hal_command)
{
    if (mode_ <= kPreemptModeStopSubmission) return;

    auto kernel_command
        = std::dynamic_pointer_cast<CudaKernelLaunchCommand>(hal_command);
    if (kernel_command == nullptr) return;
    
    if (mode_ == kPreemptModeInterrupt) {
        // TODO: assign kernel_command->can_be_killed_
        kernel_command->can_be_killed_ = true;
    }

    preempt_manager_->InstrumentKernel(kernel_command);
}

void CudaQueue::HalSynchronize()
{
    CUDA_ASSERT(Driver::StreamSynchronize(stream_));
}

void CudaQueue::HalSubmit(std::shared_ptr<HalCommand> hal_command)
{
    auto cuda_command = std::dynamic_pointer_cast<CudaCommand>(hal_command);
    XASSERT(cuda_command != nullptr, "hal_command is not a CudaCommand");

    auto kernel_command
        = std::dynamic_pointer_cast<CudaKernelLaunchCommand>(cuda_command);
    
    if (kernel_command != nullptr) {
        preempt_manager_->LaunchKernel(stream_, kernel_command);
    } else {
        CUDA_ASSERT(cuda_command->EnqueueWrapper(stream_));
    }
}

void CudaQueue::Deactivate()
{
    XASSERT(mode_ >= kPreemptModeDeactivate,
            "Deactivate() not supportted on current mode");
    preempt_manager_->Deactivate();
}

void CudaQueue::Reactivate(const CommandLog &log)
{
    XASSERT(mode_ >= kPreemptModeDeactivate,
            "Reactivate() not supportted on current mode");
    this->HalSynchronize();

    int64_t resume_command_idx = preempt_manager_->Reactivate();
    if (resume_command_idx < 0) return;

    for (auto cmd : log) {
        if (cmd->GetIdx() < resume_command_idx) continue;
        this->HalSubmit(cmd);
    }
}

void CudaQueue::Interrupt()
{
    XASSERT(mode_ >= kPreemptModeInterrupt,
            "Interrupt() not supportted on current mode");
    preempt_manager_->Interrupt();
}

CUresult CudaQueue::LaunchKernelNormal(
        std::shared_ptr<CudaKernelLaunchCommand> kernel, CUstream stream)
{
    CUcontext context;
    CUDA_ASSERT(Driver::CtxGetCurrent(&context));
    auto preempt_context = PreemptContext::GetPreemptContext(context);
    return preempt_context->DefaultLaunchKernel(stream, kernel);
}
