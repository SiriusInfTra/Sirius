#ifndef COLSERVE_CUDA_ALLOCATOR_H
#define COLSERVE_CUDA_ALLOCATOR_H

#include <mutex>
#include <map>
#include <set>
#include <memory>
#include <atomic>
#include <iostream>
#include <cstddef>
#include <cuda_runtime_api.h>

#include "mempool.hpp"

namespace colserve {
namespace sta {
class CUDAMemPool {
 public:
  using PoolEntry = CUDAMemPoolImpl::PoolEntry;
  static void Init(std::size_t nbytes);
  static CUDAMemPool* Get();

  static std::shared_ptr<PoolEntry> RawAlloc(size_t nbytes);

  CUDAMemPool(std::size_t nbytes);
  std::shared_ptr<PoolEntry> Alloc(std::size_t nbytes);
  std::shared_ptr<PoolEntry> Resize(std::shared_ptr<PoolEntry> entry, std::size_t nbytes);
  void CopyFromTo(std::shared_ptr<PoolEntry> src, std::shared_ptr<PoolEntry> dst);

 private:
  static std::unique_ptr<CUDAMemPool> cuda_mem_pool_;
  cudaStream_t stream_;
  CUDAMemPoolImpl *impl_;
};



}
}

#endif