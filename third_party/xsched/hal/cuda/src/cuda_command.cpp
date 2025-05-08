#include <cstring>
#include <cuxtra/cuxtra.h>

#include "utils/xassert.h"
#include "hal/cuda/event_pool.h"
#include "hal/cuda/cuda_assert.h"
#include "hal/cuda/cuda_command.h"
#include "preempt/xqueue/xtype.h"
#include "preempt/xqueue/xqueue.h"

#include "shim/cuda/extra.h"

using namespace xsched::hal::cuda;

std::function<void*()> CudaKernelLaunchCommand::cuda_kernel_launch_pre_hook = nullptr;

CudaCommand::~CudaCommand()
{
    if (following_event_ == nullptr) return;
    g_event_pool.Push(following_event_);
}

CUresult CudaCommand::EnqueueWrapper(CUstream stream)
{
    CUresult ret = Enqueue(stream);
    if (UNLIKELY(ret != CUDA_SUCCESS)) return ret;
    if (following_event_ != nullptr) {
        ret = Driver::EventRecord(following_event_, stream);
    }
    return ret;
}

void CudaCommand::HalSynchronize()
{
    XASSERT(following_event_ != nullptr, "following_event_ is nullptr");
    CUDA_ASSERT(Driver::EventSynchronize(following_event_));
}

bool CudaCommand::HalSynchronizable()
{
    return following_event_ != nullptr;
}

bool CudaCommand::EnableHalSynchronization()
{
    following_event_ = (CUevent)g_event_pool.Pop();
    return following_event_ != nullptr;
}

CudaKernelLaunchCommand::
CudaKernelLaunchCommand(CUfunction function,
                        unsigned int grid_dim_x,
                        unsigned int grid_dim_y,
                        unsigned int grid_dim_z,
                        unsigned int block_dim_x,
                        unsigned int block_dim_y,
                        unsigned int block_dim_z,
                        unsigned int shared_mem_bytes,
                        void **kernel_params,
                        void **extra_params,
                        bool copy_param)
    : CudaCommand(preempt::kHalCommandTypeDeactivatable)
    , function_(function)
    , grid_dim_x_(grid_dim_x)
    , grid_dim_y_(grid_dim_y)
    , grid_dim_z_(grid_dim_z)
    , block_dim_x_(block_dim_x)
    , block_dim_y_(block_dim_y)
    , block_dim_z_(block_dim_z)
    , shared_mem_bytes_(shared_mem_bytes)
    , extra_params_(extra_params)
    , kernel_params_(kernel_params)
{
    if (!copy_param) return;

    param_cnt_ = cuXtraGetParamCount(function_);
    if (param_cnt_ == 0) return;

    param_copied_ = true;
    kernel_params_ = (void **)malloc(param_cnt_ * sizeof(void *));
    // Allocate a continuous buffer for all of the params
    // buffer size = last param offset + last param size
    size_t last_offset, last_size;
    cuXtraGetParamInfo(function_, param_cnt_ - 1,
        &last_offset, &last_size, nullptr);
    size_t buffer_size = last_offset + last_size;
    param_data_ = (char *)malloc(buffer_size);

    for (uint32_t i = 0; i < param_cnt_; ++ i) {
        size_t offset, size;
        cuXtraGetParamInfo(function_, i, &offset, &size, nullptr);
        kernel_params_[i] = (void*)&param_data_[offset];
        memcpy(kernel_params_[i], kernel_params[i], size);
    }
}

CudaKernelLaunchCommand::~CudaKernelLaunchCommand()
{
    if (!param_copied_) return;

    free(param_data_);
    free(kernel_params_);
}

CUresult CudaKernelLaunchCommand::Enqueue(CUstream stream)
{
    return Driver::LaunchKernel(function_,
                                grid_dim_x_,
                                grid_dim_y_,
                                grid_dim_z_,
                                block_dim_x_,
                                block_dim_y_,
                                block_dim_z_,
                                shared_mem_bytes_,
                                stream,
                                kernel_params_,
                                extra_params_);
}

void CudaKernelLaunchCommand::BeforeHalSubmit() {
    if (CudaKernelLaunchCommand::cuda_kernel_launch_pre_hook != nullptr) {
        CudaKernelLaunchCommand::cuda_kernel_launch_pre_hook();
    }

    // if (shim::cuda::extra::IsNcclFunc(function_)) {
    //     XINFO("HalSubmit Nccl kernel %s %p", shim::cuda::extra::GetFuncName(function_), this);
    // }
}


bool CudaKernelLaunchCommand::RegisterCudaKernelLaunchPreHook(
        std::function<void*()> fn) {
    if (cuda_kernel_launch_pre_hook != nullptr) {
        XWARN("Cuda kernel launch pre hook already registered");
        return false;
    }
    cuda_kernel_launch_pre_hook = fn;
    return true;
}


CudaEventRecordCommand::CudaEventRecordCommand(CUevent event)
    : CudaCommand(preempt::kHalCommandTypeIdempotent), event_(event)
{
    XASSERT(event_ != nullptr, "cuda event should not be nullptr");
}

CudaEventRecordCommand::~CudaEventRecordCommand()
{
    if (event_ == nullptr || (! destroy_event_)) return;
    CUDA_ASSERT(Driver::EventDestroy(event_));
}

void CudaEventRecordCommand::Synchronize()
{
    auto xqueue = GetXQueue();
    
    if (xqueue == nullptr) {
        HalSynchronize();
        return;
    }

    xqueue->Synchronize(
        std::static_pointer_cast<CudaCommand>(shared_from_this()));
}

void CudaEventRecordCommand::HalSynchronize()
{
    CUDA_ASSERT(Driver::EventSynchronize(event_));
}

CUresult CudaEventRecordCommand::Enqueue(CUstream stream)
{
    return Driver::EventRecord(event_, stream);
}

CudaEventRecordWithFlagsCommand::
CudaEventRecordWithFlagsCommand(CUevent event, unsigned int flags)
    : CudaEventRecordCommand(event)
    , flags_(flags)
{

}

CUresult CudaEventRecordWithFlagsCommand::Enqueue(CUstream stream)
{
    return Driver::EventRecordWithFlags(event_, stream, flags_);
}

CudaEventWaitCommand::CudaEventWaitCommand(CUevent event,
                                           unsigned int flags)
    : CudaCommand(preempt::kHalCommandTypeIdempotent)
    , event_(event)
    , event_record_command_(nullptr)
    , flags_(flags)
{

}

CudaEventWaitCommand::CudaEventWaitCommand(
        std::shared_ptr<CudaEventRecordCommand> event_record_command,
        unsigned int flags)
    : CudaCommand(preempt::kHalCommandTypeIdempotent)
    , event_(nullptr)
    , event_record_command_(event_record_command)
    , flags_(flags)
{

}

void CudaEventWaitCommand::BeforeHalSubmit()
{
    if (event_record_command_) {
        event_record_command_->Synchronize();
    }
}

CUresult CudaEventWaitCommand::Enqueue(CUstream stream)
{
    if (event_) {
        return Driver::StreamWaitEvent(stream, event_, flags_);
    } else {
        return CUDA_SUCCESS;
    }
}
