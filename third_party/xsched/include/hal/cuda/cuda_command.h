#pragma once

#include <queue>
#include <mutex>
#include <functional>
#include <unordered_set>

#include "hal/cuda/cuda.h"
#include "hal/cuda/driver.h"
#include "preempt/hal/def.h"
#include "preempt/hal/hal_command.h"

namespace xsched::hal::cuda
{

class CudaCommand : public preempt::HalCommand
{
public:
    CudaCommand(preempt::HalCommandType hal_type): HalCommand(hal_type) {}
    virtual ~CudaCommand();

    CUresult EnqueueWrapper(CUstream stream);

    virtual void HalSynchronize() override;
    virtual bool HalSynchronizable() override;
    virtual bool EnableHalSynchronization() override;

private:
    CUevent following_event_ = nullptr;

    virtual CUresult Enqueue(CUstream stream) = 0;
};

class CudaKernelLaunchCommand : public CudaCommand
{
public:
    const CUfunction function_;
    const unsigned int grid_dim_x_;
    const unsigned int grid_dim_y_;
    const unsigned int grid_dim_z_;
    const unsigned int block_dim_x_;
    const unsigned int block_dim_y_;
    const unsigned int block_dim_z_;
    const unsigned int shared_mem_bytes_;
    void ** const extra_params_;

    bool can_be_killed_ = false;
    CUdeviceptr original_entry_point_ = 0;
    CUdeviceptr instrumented_entry_point_ = 0;

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
                            bool copy_param);
    virtual ~CudaKernelLaunchCommand();

    virtual void BeforeHalSubmit() override;

    static bool RegisterCudaKernelLaunchPreHook(std::function<void*()> fn);
    static std::function<void*()> cuda_kernel_launch_pre_hook;

private:
    void **kernel_params_ = nullptr;
    bool param_copied_ = false;
    size_t param_cnt_ = 0;
    char *param_data_ = nullptr;

    virtual CUresult Enqueue(CUstream stream) override;
};

class CudaHostFuncCommand : public CudaCommand
{
public:
    CudaHostFuncCommand(CUhostFn fn, void *user_data)
        : CudaCommand(preempt::kHalCommandTypeNormal)
        , fn_(fn), user_data_(user_data) {}
    virtual ~CudaHostFuncCommand() = default;

private:
    const CUhostFn fn_;
    void * const user_data_;

    virtual CUresult Enqueue(CUstream stream) override
    {
        return Driver::LaunchHostFunc(stream, fn_, user_data_);
    }
};

class CudaMemoryCommand : public CudaCommand
{
public:
    CudaMemoryCommand(): CudaCommand(preempt::kHalCommandTypeNormal) {}
    virtual ~CudaMemoryCommand() = default;
};

DEFINE_HAL_COMMAND3(CudaMemcpyHtoDV2Command, CudaMemoryCommand,
                    CUresult, CUstream, false,
                    Driver::MemcpyHtoDAsyncV2,
                    CUdeviceptr , dst_device, false,
                    const void *, src_host  , false,
                    size_t      , byte_count, false);

DEFINE_HAL_COMMAND3(CudaMemcpyDtoHV2Command, CudaMemoryCommand,
                    CUresult, CUstream, false,
                    Driver::MemcpyDtoHAsyncV2,
                    void *     , dst_host  , false,
                    CUdeviceptr, src_device, false,
                    size_t     , byte_count, false);

DEFINE_HAL_COMMAND3(CudaMemcpyDtoDV2Command, CudaMemoryCommand,
                    CUresult, CUstream, false,
                    Driver::MemcpyDtoDAsyncV2,
                    CUdeviceptr, dst_device, false,
                    CUdeviceptr, src_device, false,
                    size_t     , byte_count, false);

DEFINE_HAL_COMMAND1(CudaMemcpy2DV2Command, CudaMemoryCommand,
                    CUresult, CUstream, false,
                    Driver::Memcpy2DAsyncV2,
                    const CUDA_MEMCPY2D *, p_copy, true);

DEFINE_HAL_COMMAND1(CudaMemcpy3DV2Command, CudaMemoryCommand,
                    CUresult, CUstream, false,
                    Driver::Memcpy3DAsyncV2,
                    const CUDA_MEMCPY3D *, p_copy, true);

DEFINE_HAL_COMMAND3(CudaMemsetD8Command, CudaMemoryCommand,
                    CUresult, CUstream, false,
                    Driver::MemsetD8Async,
                    CUdeviceptr  , dst_device   , false,
                    unsigned char, unsigned_char, false,
                    size_t       , n            , false);

DEFINE_HAL_COMMAND3(CudaMemsetD16Command, CudaMemoryCommand,
                    CUresult, CUstream, false,
                    Driver::MemsetD16Async,
                    CUdeviceptr   , dst_device    , false,
                    unsigned short, unsigned_short, false,
                    size_t        , n             , false);

DEFINE_HAL_COMMAND3(CudaMemsetD32Command, CudaMemoryCommand,
                    CUresult, CUstream, false,
                    Driver::MemsetD32Async,
                    CUdeviceptr , dst_device  , false,
                    unsigned int, unsigned_int, false,
                    size_t      , n           , false);

DEFINE_HAL_COMMAND5(CudaMemsetD2D8Command, CudaMemoryCommand,
                    CUresult, CUstream, false,
                    Driver::MemsetD2D8Async,
                    CUdeviceptr  , dst_device   , false,
                    size_t       , dst_pitch    , false,
                    unsigned char, unsigned_char, false,
                    size_t       , width        , false,
                    size_t       , height       , false);

DEFINE_HAL_COMMAND5(CudaMemsetD2D16Command, CudaMemoryCommand,
                    CUresult, CUstream, false,
                    Driver::MemsetD2D16Async,
                    CUdeviceptr   , dst_device    , false,
                    size_t        , dst_pitch     , false,
                    unsigned short, unsigned_short, false,
                    size_t        , width         , false,
                    size_t        , height        , false);

DEFINE_HAL_COMMAND5(CudaMemsetD2D32Command, CudaMemoryCommand,
                    CUresult, CUstream, false,
                    Driver::MemsetD2D32Async,
                    CUdeviceptr , dst_device  , false,
                    size_t      , dst_pitch   , false,
                    unsigned int, unsigned_int, false,
                    size_t      , width       , false,
                    size_t      , height      , false);

DEFINE_HAL_COMMAND2(CudaMemoryAllocCommand, CudaMemoryCommand,
                    CUresult, CUstream, false,
                    Driver::MemAllocAsync,
                    CUdeviceptr *, dptr     , false,
                    size_t       , byte_size, false);

DEFINE_HAL_COMMAND1(CudaMemoryFreeCommand, CudaMemoryCommand,
                    CUresult, CUstream, false,
                    Driver::MemFreeAsync,
                    CUdeviceptr, dptr, false);

class CudaEventRecordCommand : public CudaCommand
{
public:
    CudaEventRecordCommand(CUevent event);
    virtual ~CudaEventRecordCommand();

    virtual void Synchronize() override;
    virtual void HalSynchronize() override;
    virtual bool HalSynchronizable() override { return true; }
    virtual bool EnableHalSynchronization() override { return true; }

    // Mark the event_ as destroyed, so that the event_ will be destroyed
    // in the destructor.
    void DestroyEvent() { destroy_event_ = true; }

protected:
    CUevent event_;

private:
    bool destroy_event_ = false;

    virtual CUresult Enqueue(CUstream stream) override;
};

class CudaEventRecordWithFlagsCommand : public CudaEventRecordCommand
{
public:
    CudaEventRecordWithFlagsCommand(CUevent event, unsigned int flags);
    virtual ~CudaEventRecordWithFlagsCommand() = default;

private:
    const unsigned int flags_;

    virtual CUresult Enqueue(CUstream stream) override;
};

class CudaEventWaitCommand : public CudaCommand
{
public:
    CudaEventWaitCommand(CUevent event, unsigned int flags);
    CudaEventWaitCommand(
        std::shared_ptr<CudaEventRecordCommand> event_record_command,
        unsigned int flags);
    
    virtual ~CudaEventWaitCommand() = default;

    virtual void BeforeHalSubmit() override;

private:
    const CUevent event_;
    const std::shared_ptr<CudaEventRecordCommand> event_record_command_;
    const unsigned int flags_;

    virtual CUresult Enqueue(CUstream stream) override;
};

} // namespace xsched::hal::cuda
