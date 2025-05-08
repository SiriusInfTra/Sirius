#pragma once

#include <list>
#include <mutex>
#include <cuxtra/cuxtra.h>

#include "hal/cuda/cuda.h"

#define VM_DEFAULT_SIZE     (1UL << 30)     // 1G
#define BUFFER_DEFAULT_SIZE (16UL << 20)    // 16M

namespace xsched::hal::cuda
{

class ResizableBuffer
{
public:
    ResizableBuffer(CUstream op_stream, size_t size, bool persistent=false);
    virtual ~ResizableBuffer();

    size_t Size() { return size_; }
    CUdeviceptr DevPtr() { return dev_ptr_; }
    void ExpandTo(size_t new_size);

    // whether the buffer is persistent and will not
    // be freed until the program exits
    const bool persistent_;

private:
    size_t size_;
    CUdeviceptr dev_ptr_;
    const CUstream stream_;

    size_t granularity_;
    CUmemAccessDesc rw_desc_;
    CUmemAllocationProp prop_;

    struct AllocationHandle
    {
        size_t size;
        CUmemGenericAllocationHandle handle;
    };
    std::list<AllocationHandle> handles_;
};


class InstrMemAllocator
{
public:
    InstrMemAllocator(CUstream op_stream, bool persistent=false);
    virtual ~InstrMemAllocator();

    /// @brief Allocate an instruction device memory block of input size.
    /// @param size the size of the instruction device memory block
    /// @return the virtual address of the instruction device memory block
    CUdeviceptr Alloc(size_t size);

    // whether the buffer is persistent and will not
    // be freed until the program exits
    const bool persistent_;

private:
    std::mutex mtx_;
    size_t used_size_ = 0;
    size_t total_size_;
    size_t granularity_;

    CUdevice device_;
    CUcontext context_;
    const CUstream stream_;
    CUdeviceptr block_base_;
};

} // namespace xsched::hal::cuda
