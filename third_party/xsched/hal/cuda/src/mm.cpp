#include <cstring>
#include <cuxtra/cuxtra.h>

#include "utils/xassert.h"
#include "hal/cuda/mm.h"
#include "hal/cuda/driver.h"
#include "hal/cuda/cuda_assert.h"

using namespace xsched::hal::cuda;

ResizableBuffer::ResizableBuffer(CUstream op_stream,
                                 size_t size,
                                 bool persistent)
    : persistent_(persistent), stream_(op_stream)
{
    CUdevice device;
    CUDA_ASSERT(Driver::CtxGetDevice(&device));
    prop_ = {
        .type = CU_MEM_ALLOCATION_TYPE_PINNED,
        .requestedHandleTypes = CUmemAllocationHandleType_enum(0),
        .location = {
            .type = CU_MEM_LOCATION_TYPE_DEVICE,
            .id = device,
        },
        .win32HandleMetaData = nullptr,
        .reserved = 0,
    };
    rw_desc_ = {
        .location = {
            .type = CU_MEM_LOCATION_TYPE_DEVICE,
            .id = device,
        },
        .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
    };

    // alloc vm space
    CUDA_ASSERT(Driver::MemAddressReserve(
        &dev_ptr_, VM_DEFAULT_SIZE, 0, 0, 0));
    CUDA_ASSERT(Driver::MemGetAllocationGranularity(
        &granularity_, &prop_, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    
    // alloc pm space
    CUmemGenericAllocationHandle cu_handle;
    size_ = ROUND_UP(size, granularity_);
    CUDA_ASSERT(Driver::MemCreate(&cu_handle, size_, &prop_, 0));
    handles_.emplace_back(AllocationHandle{.size = size_,
                                           .handle = cu_handle});

    // map vm to pm
    CUDA_ASSERT(Driver::MemMap(dev_ptr_, size_, 0, cu_handle, 0));
    CUDA_ASSERT(Driver::MemSetAccess(dev_ptr_, size_, &rw_desc_, 1));

    // clear buffer
    CUDA_ASSERT(Driver::MemsetD8Async(dev_ptr_, 0, size_, stream_));
    CUDA_ASSERT(Driver::StreamSynchronize(stream_));
}

ResizableBuffer::~ResizableBuffer()
{
    if (persistent_) return;
    CUDA_ASSERT(Driver::MemUnmap(dev_ptr_, size_));
    for (auto h : handles_) CUDA_ASSERT(Driver::MemRelease(h.handle));
    CUDA_ASSERT(Driver::MemAddressFree(dev_ptr_, VM_DEFAULT_SIZE));
}

void ResizableBuffer::ExpandTo(size_t new_size)
{
    if (new_size <= size_) return;
    if (new_size > VM_DEFAULT_SIZE) {
        XASSERT(false,
                "resizable buffer %p cannot be expanded to %ldB: "
                "exceeds max size of %ldB",
                (void *)dev_ptr_, new_size, VM_DEFAULT_SIZE);
        return;
    }
    
    XASSERT(false, "expand not supported yet");

    new_size = ROUND_UP(new_size, granularity_);
    size_t handle_size = new_size - size_;

    // alloc pm space
    CUmemGenericAllocationHandle cu_handle;
    CUDA_ASSERT(Driver::MemCreate(&cu_handle, handle_size, &prop_, 0));
    handles_.emplace_back(AllocationHandle{.size = handle_size,
                                           .handle = cu_handle});

    // map vm to pm
    CUDA_ASSERT(Driver::MemMap(
        dev_ptr_ + size_, handle_size, 0, cu_handle, 0));
    CUDA_ASSERT(Driver::MemSetAccess(
        dev_ptr_ + size_, handle_size, &rw_desc_, 1));

    // clear buffer
    CUDA_ASSERT(Driver::MemsetD8Async(
        dev_ptr_ + size_, 0, handle_size, stream_));
    CUDA_ASSERT(Driver::StreamSynchronize(stream_));

    size_ = new_size;
}

InstrMemAllocator::InstrMemAllocator(CUstream op_stream, bool persistent)
    : persistent_(persistent), stream_(op_stream)
{
    CUDA_ASSERT(Driver::CtxGetDevice(&device_));
    CUDA_ASSERT(Driver::StreamGetCtx(stream_, &context_));

    CUmemAllocationProp prop {
        .type = CU_MEM_ALLOCATION_TYPE_PINNED,
        .requestedHandleTypes = CUmemAllocationHandleType_enum(0),
        .location = {
            .type = CU_MEM_LOCATION_TYPE_DEVICE,
            .id = device_,
        },
        .win32HandleMetaData = nullptr,
        .reserved = 0,
    };
    CUDA_ASSERT(Driver::MemGetAllocationGranularity(
        &granularity_, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

    total_size_ = ROUND_UP(BUFFER_DEFAULT_SIZE, granularity_);
    block_base_ = cuXtraInstrMemBlockAlloc(context_, total_size_);
}

InstrMemAllocator::~InstrMemAllocator()
{
    if (persistent_) return;
    cuXtraInstrMemBlockFree(context_, block_base_);
}

CUdeviceptr InstrMemAllocator::Alloc(size_t size)
{
    std::unique_lock<std::mutex> lock(mtx_);

    size_t offset = used_size_;
    used_size_ += size;
    if (used_size_ <= total_size_) return block_base_ + offset;

    XASSERT(false, "expand not supported yet");

    // the buffer needs to be expanded
    size_t old_size = total_size_;
    total_size_ = ROUND_UP(used_size_, granularity_);

    CUdeviceptr old_base = block_base_;
    block_base_ = cuXtraInstrMemBlockAlloc(context_, total_size_);
    cuXtraInstrMemcpyDtoD(block_base_, old_base, old_size, stream_);
    cuXtraInstrMemBlockFree(context_, old_base);

    return block_base_ + offset;
}
