#pragma once

#include "hal/cudla/driver.h"
#include "preempt/hal/hal_command.h"

namespace xsched::hal::cudla
{

class CudlaCommand : public preempt::HalCommand
{
public:
    CudlaCommand(preempt::HalCommandType hal_type): HalCommand(hal_type) {}
    virtual ~CudlaCommand();

    cudaError_t EnqueueWrapper(cudaStream_t stream);

    virtual void HalSynchronize() override;
    virtual bool HalSynchronizable() override;
    virtual bool EnableHalSynchronization() override;

private:
    cudaEvent_t following_event_ = nullptr;

    virtual cudaError_t Enqueue(cudaStream_t stream) = 0;
};

class CudlaTaskCommand : public CudlaCommand
{
public:
    CudlaTaskCommand(cudlaDevHandle const dev_handle,
                     const cudlaTask * const tasks,
                     uint32_t const num_tasks,
                     uint32_t const flags);
    virtual ~CudlaTaskCommand();

private:
    cudlaDevHandle const dev_handle_;
    cudlaTask *tasks_;
    uint32_t const num_tasks_;
    uint32_t const flags_;

    virtual cudaError_t Enqueue(cudaStream_t stream) override;
};

class CudlaMemoryCommand : public CudlaCommand
{
public:
    CudlaMemoryCommand(void *dst, const void *src,
                       size_t size, cudaMemcpyKind kind);
    virtual ~CudlaMemoryCommand() = default;

private:
    void *dst_;
    const void *src_;
    size_t size_;
    cudaMemcpyKind kind_;

    virtual cudaError_t Enqueue(cudaStream_t stream) override;
};

class CudlaEventRecordCommand : public CudlaCommand
{
public:
    CudlaEventRecordCommand(cudaEvent_t event);
    virtual ~CudlaEventRecordCommand() = default;

    virtual void HalSynchronize() override;
    virtual bool HalSynchronizable() override { return true; }
    virtual bool EnableHalSynchronization() override { return true; }

private:
    cudaEvent_t event_;

    virtual cudaError_t Enqueue(cudaStream_t stream) override;
};

} // namespace xsched::hal::cudla
