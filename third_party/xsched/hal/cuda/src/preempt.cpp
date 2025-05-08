#include <unordered_map>
#include <cuxtra/cuxtra.h>

#include "hal/cuda/preempt.h"
#include "hal/cuda/cuda_assert.h"

#define MAX_BLOCK_SIZE          2048
#define PREEMPT_BUFFER_DEBUG    false

using namespace xsched::preempt;
using namespace xsched::hal::cuda;

PreemptContext::PreemptContext(CUcontext context)
    : context_(context), interrupt_thread_running_(false)
{
    CUDA_ASSERT(Driver::CtxGetDevice(&device_));
    
    // create operation stream with greatest priority
    int least_priority, greatest_priority;
    CUDA_ASSERT(Driver::CtxGetStreamPriorityRange(
        &least_priority, &greatest_priority));
    CUDA_ASSERT(Driver::StreamCreateWithPriority(
        &operation_stream_, 0, greatest_priority));
    
    trap_manager_ = std::make_unique<TrapManager>(
        device_, context_, operation_stream_);
    preempt_buffer_ = std::make_unique<ResizableBuffer>(
        operation_stream_, BUFFER_DEFAULT_SIZE, true);
    CUDA_ASSERT(Driver::MemsetD8Async(preempt_buffer_->DevPtr(), 0,
        BUFFER_DEFAULT_SIZE, operation_stream_));
    CUDA_ASSERT(Driver::StreamSynchronize(operation_stream_));

    instrument_manager_ = std::make_shared<InstrumentManager>(
        device_, context_, operation_stream_);
}

PreemptContext::~PreemptContext()
{
    // PreemptContext is held by a static map. It will be
    // deconstructed when the process exits. So we don't
    // need to destroy the operation stream here.

    // CUDA_ASSERT(Driver::StreamDestroy(operation_stream_));
    
    if (!interrupt_thread_running_.load()) return;

    interrupt_thread_running_.store(false);
    interrupt_mutex_.lock();
    do_interrupt_ = true;
    interrupt_mutex_.unlock();
    interrupt_cv_.notify_all();
    interrupt_thread_->join();
}

std::shared_ptr<PreemptContext> PreemptContext::GetPreemptContext(CUcontext context)
{
    static std::mutex map_mutex;
    static std::unordered_map<CUcontext, std::shared_ptr<PreemptContext>>
        context_map;
    
    std::unique_lock<std::mutex> lock(map_mutex);
    auto it = context_map.find(context);
    if (it != context_map.end()) return it->second;
    
    auto preempt_context = std::make_shared<PreemptContext>(context);
    context_map[context] = preempt_context;
    return preempt_context;
}

void PreemptContext::InitializeTrap()
{
    static std::mutex init_mutex;
    static bool initialized = false;

    std::unique_lock<std::mutex> lock(init_mutex);
    if (initialized) return;

    trap_manager_->SetTrapHandler();
    initialized = true;

    // interrupt_thread_running_.store(true);
    // interrupt_thread_ = std::make_unique<std::thread>([&]() {
    //     while (true) {
    //         std::unique_lock<std::mutex> lock(interrupt_mutex_);
    //         while (!do_interrupt_) { interrupt_cv_.wait(lock); }
    //         do_interrupt_ = false;
    //         lock.unlock();

    //         if (!interrupt_thread_running_.load()) return;

    //         trap_manager_->InterruptContext();
    //     }
    // });
}

void PreemptContext::InterruptContext()
{
    trap_manager_->InterruptContext();

    // interrupt_mutex_.lock();
    // do_interrupt_ = true;
    // interrupt_mutex_.unlock();

    // interrupt_cv_.notify_all();
}

CUdevice PreemptContext::GetDevice() const
{
    return device_;
}

CUcontext PreemptContext::GetContext() const
{
    return context_;
}

CUstream PreemptContext::GetOperationStream() const
{
    return operation_stream_;
}

std::unique_lock<std::mutex> PreemptContext::GetLock()
{
    return std::move(std::unique_lock<std::mutex>(mutex_));
}

std::shared_ptr<InstrumentManager> PreemptContext::GetInstrumentManager()
{
    return instrument_manager_;
}

CUresult PreemptContext::DefaultLaunchKernel(CUstream stream,
    std::shared_ptr<CudaKernelLaunchCommand> kernel)
{
    char args_buffer[28];
    *(uint64_t *)(args_buffer +  0) = preempt_buffer_->DevPtr();
    *(uint64_t *)(args_buffer +  8) = 0;
    *(uint64_t *)(args_buffer + 16) = -1;
    *(uint32_t *)(args_buffer + 24) = false;

    std::unique_lock<std::mutex> lock(mutex_);

    cuXtraSetDebuggerParams(kernel->function_, args_buffer, sizeof(args_buffer));
    return kernel->EnqueueWrapper(stream);
}

PreemptManager::PreemptManager(XPreemptMode mode,
                               CUcontext context,
                               CUstream stream)
    : mode_(mode)
    , stream_(stream)
{
    preempt_context_ = PreemptContext::GetPreemptContext(context);

    if (mode_ >= kPreemptModeInterrupt) {
        preempt_context_->InitializeTrap();
    }

    if (mode_ >= kPreemptModeDeactivate) {
        preempt_buffer_ = std::make_unique<ResizableBuffer>(
            preempt_context_->GetOperationStream(),
            BUFFER_DEFAULT_SIZE, true);
    }
}

PreemptManager::~PreemptManager()
{
    
}

void PreemptManager::InstrumentKernel(
    std::shared_ptr<CudaKernelLaunchCommand> kernel)
{
    size_t block_cnt = kernel->grid_dim_x_ *
                       kernel->grid_dim_y_ *
                       kernel->grid_dim_z_;
    size_t block_buffer_size
        = block_cnt * 2 * sizeof(uint32_t) + 2 * sizeof(uint64_t);
    preempt_buffer_->ExpandTo(block_buffer_size);

    CUdeviceptr original_entry_point, instrumented_entry_point;
    auto instrument_manager = preempt_context_->GetInstrumentManager();
    instrument_manager->GetInstrumentedKernel(kernel->function_,
                                              original_entry_point,
                                              instrumented_entry_point);
    
    if (original_entry_point != 0 && instrumented_entry_point != 0) {
        // the kernel has been instrumented
        kernel->original_entry_point_ = original_entry_point;
        kernel->instrumented_entry_point_ = instrumented_entry_point;
        return;
    }
    
    // the kernel has not been instrumented, instrument it
    auto context_lock = preempt_context_->GetLock();
    original_entry_point = cuXtraGetEntryPoint(kernel->function_);
    context_lock.unlock();

    instrument_manager->InstrumentKernel(kernel->function_,
                                         original_entry_point,
                                         instrumented_entry_point);
    kernel->original_entry_point_ = original_entry_point;
    kernel->instrumented_entry_point_ = instrumented_entry_point;
}

void PreemptManager::LaunchKernel(CUstream stream,
    std::shared_ptr<CudaKernelLaunchCommand> kernel)
{
    if (mode_ <= kPreemptModeStopSubmission) {
        preempt_context_->DefaultLaunchKernel(stream, kernel);
        return;
    }

    char args_buffer[28];
    *(uint64_t *)(args_buffer +  0) = preempt_buffer_->DevPtr();
    *(uint64_t *)(args_buffer +  8) = kernel->instrumented_entry_point_;
    *(uint64_t *)(args_buffer + 16) = kernel->GetIdx();
    *(uint32_t *)(args_buffer + 24) = kernel->can_be_killed_;

    CUdeviceptr func_entry_point
        = first_preempted_idx_ == kernel->GetIdx()
        ? preempt_context_->GetInstrumentManager()->GetResumeEntryPoint()
        : kernel->instrumented_entry_point_;

    auto context_lock = preempt_context_->GetLock();

    cuXtraSetDebuggerParams(kernel->function_, args_buffer, sizeof(args_buffer));
    
    cuXtraSetEntryPoint(kernel->function_, func_entry_point);
    kernel->EnqueueWrapper(stream);
    cuXtraSetEntryPoint(kernel->function_, kernel->original_entry_point_);
}

void PreemptManager::Deactivate()
{
    CUDA_ASSERT(Driver::MemsetD8Async(preempt_buffer_->DevPtr(),
        1, sizeof(uint64_t), preempt_context_->GetOperationStream()));
}

int64_t PreemptManager::Reactivate()
{
    int64_t preempt_idx = 0;
    CUDA_ASSERT(Driver::MemcpyDtoHAsyncV2(&preempt_idx,
        preempt_buffer_->DevPtr() + sizeof(uint64_t),
        sizeof(int64_t), preempt_context_->GetOperationStream()));
    CUDA_ASSERT(Driver::MemsetD8Async(preempt_buffer_->DevPtr(), 0,
        sizeof(uint64_t) + sizeof(int64_t),
        preempt_context_->GetOperationStream()));
    CUDA_ASSERT(Driver::StreamSynchronize(
        preempt_context_->GetOperationStream()));
    preempt_idx = preempt_idx == 0 ? -1 : preempt_idx;

    #if PREEMPT_BUFFER_DEBUG
        printf("preempt idx: %ld\n", preempt_idx);
        uint32_t bufferHost[MAX_BLOCK_SIZE * 2 + 4];
        CUDA_ASSERT(Driver::MemcpyDtoHAsyncV2(bufferHost,
            preempt_buffer_->DevPtr(), sizeof(bufferHost),
            preempt_context_->GetOperationStream()));
        CUDA_ASSERT(Driver::StreamSynchronize(
            preempt_context_->GetOperationStream()));
        for (int i = 0; i < MAX_BLOCK_SIZE; ++i) {
            printf("block %d:\t%d,\t%d\n",
                   i,
                   bufferHost[2*i+4],
                   bufferHost[2*i+5]);
        }
    #endif

    first_preempted_idx_ = preempt_idx;
    return preempt_idx;
}

void PreemptManager::Interrupt()
{
    if (mode_ <= kPreemptModeDeactivate) return;

    // FIXME: what if multiple threads call Interrupt()?
    CUDA_ASSERT(Driver::StreamSynchronize(
        preempt_context_->GetOperationStream()));

    preempt_context_->InterruptContext();
}
