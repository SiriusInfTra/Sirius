#include <mutex>
#include <unordered_map>
#include <cuxtra/cuxtra.h>

#include "utils/common.h"
#include "utils/xassert.h"
#include "hal/cuda/instrument.h"
#include "hal/cuda/cuda_assert.h"

using namespace xsched::hal::cuda;

std::mutex InstrumentManager::map_mutex;
std::unordered_map<CUfunction, InstrumentedKernel> InstrumentManager::kernel_map;

InstrumentManager::InstrumentManager(CUdevice device,
                                     CUcontext context,
                                     CUstream op_stream)
    : device_(device)
    , context_(context)
    , op_stream_(op_stream)
{
    instr_mem_allocator_
        = std::make_unique<InstrMemAllocator>(op_stream, true);
    
    size_t resume_size;
    const void *resume_instructions_host;
    GetResumeInstructions(&resume_instructions_host, &resume_size);

    resume_instructions_entry_point_
        = instr_mem_allocator_->Alloc(resume_size);

    cuXtraInstrMemcpyHtoD(resume_instructions_entry_point_,
        resume_instructions_host, resume_size, op_stream_);
}

InstrumentManager::~InstrumentManager()
{
    // TODO: release resume_instructions_obj_?
}
    
CUdeviceptr InstrumentManager::GetResumeEntryPoint() const
{
    return resume_instructions_entry_point_;
}

void InstrumentManager::InstrumentKernel(
    CUfunction function,
    CUdeviceptr original_entry_point,
    CUdeviceptr &instrumented_entry_point)
{
    std::unique_lock<std::mutex> lock(map_mutex);
    auto it = kernel_map.find(function);
    if (it != kernel_map.end()) {
        instrumented_entry_point = it->second.instrumented_entry_point;
        return;
    }

    size_t check_preempt_size;
    const void *check_preempt_instructions_host;
    GetCheckPreemptInstructions(&check_preempt_instructions_host,
                                &check_preempt_size);
    
    size_t kernel_size;
    const void *kernel_instructions_host;
    cuXtraGetBinary(context_, function, &kernel_instructions_host, &kernel_size, false);
    
    instrumented_entry_point = instr_mem_allocator_->Alloc(kernel_size + check_preempt_size);
    
    cuXtraInstrMemcpyHtoD(instrumented_entry_point, check_preempt_instructions_host,
                          check_preempt_size, op_stream_);
    cuXtraInstrMemcpyHtoD(
        instrumented_entry_point + check_preempt_size,
        kernel_instructions_host, kernel_size, op_stream_);
    
    size_t reg_cnt = cuXtraGetLocalRegsPerThread(function);
    if (reg_cnt < 32) cuXtraSetLocalRegsPerThread(function, 32);

    size_t barrier_cnt = cuXtraGetBarrierCnt(function);
    if (barrier_cnt < 1) cuXtraSetBarrierCnt(function, 1);

    cuXtraInvalInstrCache(context_);
    
    kernel_map[function] = InstrumentedKernel {
        .function = function,
        .original_entry_point = original_entry_point,
        .instrumented_entry_point = instrumented_entry_point,
    };
}

void InstrumentManager::GetInstrumentedKernel(
    CUfunction function,
    CUdeviceptr &original_entry_point,
    CUdeviceptr &instrumented_entry_point)
{
    std::unique_lock<std::mutex> lock(map_mutex);
    auto it = kernel_map.find(function);
    if (it == kernel_map.end()) {
        original_entry_point = 0;
        instrumented_entry_point = 0;
        return;
    }

    original_entry_point = it->second.original_entry_point;
    instrumented_entry_point = it->second.instrumented_entry_point;
}
