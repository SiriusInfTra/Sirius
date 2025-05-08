#pragma once

#include <memory>
#include <cuxtra/cuxtra.h>

#include "hal/cuda/cuda.h"

namespace xsched::hal::cuda
{

class TrapManager
{
public:
    TrapManager(CUdevice device,
                CUcontext context,
                CUstream operation_stream);
    ~TrapManager() = default;

    void SetTrapHandler();
    void InterruptContext();
    void DumpTrapHandler();

private:
    const CUdevice device_;
    const CUcontext context_;
    const CUstream operation_stream_;

    size_t trap_handler_size_ = 0;
    CUdeviceptr trap_handler_dev_ = 0;
};

} // namespace xsched::hal::cuda
