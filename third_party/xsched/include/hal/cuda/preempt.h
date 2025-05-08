#pragma once

#include <mutex>
#include <memory>
#include <atomic>
#include <thread>

#include "hal/cuda/mm.h"
#include "hal/cuda/cuda.h"
#include "hal/cuda/trap.h"
#include "hal/cuda/driver.h"
#include "hal/cuda/instrument.h"
#include "hal/cuda/cuda_command.h"
#include "preempt/xqueue/xtype.h"

namespace xsched::hal::cuda
{

class PreemptContext
{
public:
    PreemptContext(CUcontext context);
    ~PreemptContext();

    /// @brief Get the PreemptContext of a CUcontext. The
    ///        PreemptContext will be created if not exist.
    static std::shared_ptr<PreemptContext>
    GetPreemptContext(CUcontext context);

    void InitializeTrap();
    void InterruptContext();

    CUdevice GetDevice() const;
    CUcontext GetContext() const;
    CUstream GetOperationStream() const;

    std::unique_lock<std::mutex> GetLock();
    std::shared_ptr<InstrumentManager> GetInstrumentManager();

    CUresult DefaultLaunchKernel(CUstream stream,
        std::shared_ptr<CudaKernelLaunchCommand> kernel);

private:
    CUdevice device_;
    CUcontext context_;
    CUstream operation_stream_;

    bool do_interrupt_ = false;
    std::mutex interrupt_mutex_;
    std::condition_variable interrupt_cv_;
    std::atomic_bool interrupt_thread_running_;
    std::unique_ptr<std::thread> interrupt_thread_ = nullptr;

    std::mutex mutex_;
    std::unique_ptr<TrapManager> trap_manager_;
    std::unique_ptr<ResizableBuffer> preempt_buffer_;
    std::shared_ptr<InstrumentManager> instrument_manager_;
};

class PreemptManager
{
public:
    PreemptManager(preempt::XPreemptMode mode, CUcontext context, CUstream stream);
    ~PreemptManager();

    void InstrumentKernel(
        std::shared_ptr<CudaKernelLaunchCommand> kernel_command);
    void LaunchKernel(CUstream stream,
        std::shared_ptr<CudaKernelLaunchCommand> kernel);

    void Deactivate();
    int64_t Reactivate();
    void Interrupt();

private:
    const preempt::XPreemptMode mode_;
    const CUstream stream_;

    int64_t first_preempted_idx_ = -1;
    std::shared_ptr<PreemptContext> preempt_context_ = nullptr;
    std::unique_ptr<ResizableBuffer> preempt_buffer_ = nullptr;
};

} // namespace xsched::hal::cuda
