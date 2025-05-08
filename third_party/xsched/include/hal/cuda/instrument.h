#pragma once

#include <memory>
#include <unordered_map>

#include "utils/common.h"
#include "hal/cuda/mm.h"
#include "hal/cuda/cuda.h"

namespace xsched::hal::cuda
{

struct InstrumentedKernel
{
    CUfunction function;
    CUdeviceptr original_entry_point;
    CUdeviceptr instrumented_entry_point;
};

class InstrumentManager
{
public:
    InstrumentManager(CUdevice device,
                      CUcontext context,
                      CUstream op_stream);
    ~InstrumentManager();

    CUdeviceptr GetResumeEntryPoint() const;

    void InstrumentKernel(CUfunction function,
                          CUdeviceptr original_entry_point,
                          CUdeviceptr &instrumented_entry_point);

    void GetInstrumentedKernel(CUfunction function,
                               CUdeviceptr &original_entry_point,
                               CUdeviceptr &instrumented_entry_point);

    static void InstrumentTrapHandler(void *trap_handler_host,
                                      CUdeviceptr trap_handler_dev,
                                      size_t trap_handler_size,
                                      void *extra_instrs_host,
                                      CUdeviceptr extra_instrs_device,
                                      size_t extra_instrs_size);

private:
    const CUdevice device_;
    const CUcontext context_;
    const CUstream op_stream_;
    
    CUdeviceptr resume_instructions_entry_point_ = 0;
    std::unique_ptr<InstrMemAllocator> instr_mem_allocator_ = nullptr;

    static std::mutex map_mutex;
    static std::unordered_map<CUfunction, InstrumentedKernel> kernel_map;

    static void GetResumeInstructions(const void **instructions,
                                      size_t *size);
    static void GetCheckPreemptInstructions(const void **instructions,
                                            size_t *size);
};

} // namespace xsched::hal::cuda
