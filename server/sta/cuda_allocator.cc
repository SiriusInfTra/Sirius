#include <iostream>
#include <cuda_runtime_api.h>

#include "cuda_allocator.h"
#include <glog/logging.h>

namespace colserve {
namespace sta {

std::unique_ptr<CUDAMemPool> CUDAMemPool::cuda_mem_pool_;

CUDAMemPool* CUDAMemPool::Get() {
  if (cuda_mem_pool_ == nullptr) {
    LOG(FATAL) << "[CUDAMemPool]: CUDAMemPool not initialized";
  }
  return cuda_mem_pool_.get();
}

void CUDAMemPool::Init(std::size_t nbytes) {
  // LOG(INFO) << "[CUDA Memory Pool] initilized with size " << size / 1024 / 1024 << " Mb";
  cuda_mem_pool_ = std::make_unique<CUDAMemPool>(nbytes);
}

CUDAMemPool::CUDAMemPool(std::size_t nbytes)  {
//    remove("/dev/shm/gpu_colocation_mempool");
    CUDAMemPoolImpl::MemPoolConfig config{
        .cudaDevice = 0,
        .cudaMemorySize = nbytes,
        .sharedMemoryName = "gpu_colocation_mempool",
        .sharedMemorySize = 1024 * 1024 * 1024, /* 1G */
    };
    impl_ = new CUDAMemPoolImpl{config};
    CUDA_CALL(cudaStreamCreate(&stream_));
}


std::shared_ptr<CUDAMemPool::PoolEntry> CUDAMemPool::Alloc(std::size_t nbytes) {
    return impl_->Alloc(nbytes);
}

std::shared_ptr<CUDAMemPool::PoolEntry> CUDAMemPool::Resize(
    std::shared_ptr<PoolEntry> entry, std::size_t nbytes) {
    // TODO: handle reallocate
    entry.reset();
    return impl_->Alloc(nbytes);
}

void CUDAMemPool::CopyFromTo(std::shared_ptr<PoolEntry> src, std::shared_ptr<PoolEntry> dst) {
  if (src == nullptr || dst == nullptr) return;
  if (src->addr == dst->addr) return;

  CUDA_CALL(cudaMemcpyAsync(dst->addr, src->addr, 
      std::min(src->nbytes, dst->nbytes), cudaMemcpyDefault, stream_));
  CUDA_CALL(cudaStreamSynchronize(stream_));
}





std::shared_ptr<CUDAMemPool::PoolEntry> CUDAMemPool::RawAlloc(size_t nbytes) {
  void* ptr;
  CUDA_CALL(cudaSetDevice(0));
  CUDA_CALL(cudaMalloc(&ptr, nbytes));
  return std::shared_ptr<PoolEntry>(
    new PoolEntry{ptr, nbytes}, [](PoolEntry *entry) {
      CUDA_CALL(cudaSetDevice(0));
      CUDA_CALL(cudaFree(entry->addr));
      delete entry;
    });
}


}
}