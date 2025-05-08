#pragma once

#include <queue>
#include <mutex>

#include "cuda_preempt/cuda/def.h"
#include "xpreempt/hal/hal_command.h"

class CudaCommand : public HalCommand
{
public:
    CudaCommand() = default;
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

    virtual bool Cancelable() override;
    virtual bool WriteMemory() override;

    virtual void BeforeHalSubmit() override;

private:
    void **kernel_params_ = nullptr;
    bool param_copied_ = false;
    size_t param_cnt_ = 0;
    char *param_data_ = nullptr;

    virtual CUresult Enqueue(CUstream stream) override;
};

class CudaMemoryCommand : public CudaCommand
{
public:
    CudaMemoryCommand() = default;
    virtual ~CudaMemoryCommand() = default;

    virtual bool Cancelable() override;
    virtual bool WriteMemory() override;
};

class CudaMemcpyHtoDV2Command : public CudaMemoryCommand
{
public:
    CudaMemcpyHtoDV2Command(CUdeviceptr dst_device,
                            const void *src_host,
                            size_t byte_count);
    virtual ~CudaMemcpyHtoDV2Command() = default;

private:
    const CUdeviceptr dst_device_;
    const void * const src_host_;
    const size_t byte_count_;

    virtual CUresult Enqueue(CUstream stream) override;
};

class CudaMemcpyDtoHV2Command : public CudaMemoryCommand
{
public:
    CudaMemcpyDtoHV2Command(void *dst_host,
                            CUdeviceptr src_device,
                            size_t byte_count);
    virtual ~CudaMemcpyDtoHV2Command() = default;

private:
    void * const dst_host_;
    const CUdeviceptr src_device_;
    const size_t byte_count_;

    virtual CUresult Enqueue(CUstream stream) override;
};

class CudaMemcpyDtoDV2Command : public CudaMemoryCommand
{
public:
    CudaMemcpyDtoDV2Command(CUdeviceptr dst_device,
                            CUdeviceptr src_device,
                            size_t byte_count);
    virtual ~CudaMemcpyDtoDV2Command() = default;

private:
    const CUdeviceptr dst_device_;
    const CUdeviceptr src_device_;
    const size_t byte_count_;

    virtual CUresult Enqueue(CUstream stream) override;
};

class CudaMemcpy2DV2Command : public CudaMemoryCommand
{
public:
    CudaMemcpy2DV2Command(const CUDA_MEMCPY2D *p_copy);
    virtual ~CudaMemcpy2DV2Command() = default;

private:
    const CUDA_MEMCPY2D copy_;

    virtual CUresult Enqueue(CUstream stream) override;
};

class CudaMemcpy3DV2Command : public CudaMemoryCommand
{
public:
    CudaMemcpy3DV2Command(const CUDA_MEMCPY3D *p_copy);
    virtual ~CudaMemcpy3DV2Command() = default;

private:
    const CUDA_MEMCPY3D copy_;

    virtual CUresult Enqueue(CUstream stream) override;
};

class CudaMemsetD8Command : public CudaMemoryCommand
{
public:
    CudaMemsetD8Command(CUdeviceptr dst_device,
                        unsigned char unsigned_char,
                        size_t n);
    virtual ~CudaMemsetD8Command() = default;

private:
    const CUdeviceptr dst_device_;
    const unsigned char unsigned_char_;
    const size_t n_;

    virtual CUresult Enqueue(CUstream stream) override;
};

class CudaMemsetD16Command : public CudaMemoryCommand
{
public:
    CudaMemsetD16Command(CUdeviceptr dst_device,
                         unsigned short unsigned_short,
                         size_t n);
    virtual ~CudaMemsetD16Command() = default;

private:
    const CUdeviceptr dst_device_;
    const unsigned short unsigned_short_;
    const size_t n_;

    virtual CUresult Enqueue(CUstream stream) override;
};

class CudaMemsetD32Command : public CudaMemoryCommand
{
public:
    CudaMemsetD32Command(CUdeviceptr dst_device,
                         unsigned int unsigned_int,
                         size_t n);
    virtual ~CudaMemsetD32Command() = default;

private:
    const CUdeviceptr dst_device_;
    const unsigned int unsigned_int_;
    const size_t n_;

    virtual CUresult Enqueue(CUstream stream) override;
};

class CudaMemsetD2D8Command : public CudaMemoryCommand
{
public:
    CudaMemsetD2D8Command(CUdeviceptr dst_device,
                          size_t dst_pitch,
                          unsigned char unsigned_char,
                          size_t width,
                          size_t height);
    virtual ~CudaMemsetD2D8Command() = default;

private:
    const CUdeviceptr dst_device_;
    const size_t dst_pitch_;
    const unsigned char unsigned_char_;
    const size_t width_;
    const size_t height_;

    virtual CUresult Enqueue(CUstream stream) override;
};

class CudaMemsetD2D16Command : public CudaMemoryCommand
{
public:
    CudaMemsetD2D16Command(CUdeviceptr dst_device,
                           size_t dst_pitch,
                           unsigned short unsigned_short,
                           size_t width,
                           size_t height);
    virtual ~CudaMemsetD2D16Command() = default;

private:
    const CUdeviceptr dst_device_;
    const size_t dst_pitch_;
    const unsigned short unsigned_short_;
    const size_t width_;
    const size_t height_;

    virtual CUresult Enqueue(CUstream stream) override;
};

class CudaMemsetD2D32Command : public CudaMemoryCommand
{
public:
    CudaMemsetD2D32Command(CUdeviceptr dst_device,
                           size_t dst_pitch,
                           unsigned int unsigned_int,
                           size_t width,
                           size_t height);
    virtual ~CudaMemsetD2D32Command() = default;

private:
    const CUdeviceptr dst_device_;
    const size_t dst_pitch_;
    const unsigned int unsigned_int_;
    const size_t width_;
    const size_t height_;

    virtual CUresult Enqueue(CUstream stream) override;
};

class CudaMemoryAllocCommand : public CudaMemoryCommand
{
public:
    CudaMemoryAllocCommand(CUdeviceptr *dptr,
                           size_t byte_size);
    virtual ~CudaMemoryAllocCommand() = default;

private:
    CUdeviceptr * const dptr_;
    const size_t byte_size_;

    virtual CUresult Enqueue(CUstream stream) override;
};

class CudaMemoryFreeCommand : public CudaMemoryCommand
{
public:
    CudaMemoryFreeCommand(CUdeviceptr dptr);
    virtual ~CudaMemoryFreeCommand() = default;

private:
    const CUdeviceptr dptr_;

    virtual CUresult Enqueue(CUstream stream) override;
};

class CudaEventRecordCommand : public CudaCommand
{
public:
    CudaEventRecordCommand(CUevent event);
    virtual ~CudaEventRecordCommand() = default;

    virtual bool Cancelable() override;
    virtual bool WriteMemory() override;

    virtual void Synchronize() override;
    virtual void HalSynchronize() override;
    virtual bool HalSynchronizable() override;
    virtual bool EnableHalSynchronization() override;

protected:
    CUevent event_;

private:
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
    
    ~CudaEventWaitCommand() = default;

    virtual void BeforeHalSubmit() override;

    virtual bool Cancelable() override;
    virtual bool WriteMemory() override;

private:
    const CUevent event_;
    const std::shared_ptr<CudaEventRecordCommand> event_record_command_;
    const unsigned int flags_;

    virtual CUresult Enqueue(CUstream stream) override;
};
