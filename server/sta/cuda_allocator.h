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

namespace colserve {
namespace sta {

class CUDAMemPool {
 public:
  struct PoolEntry {
    void* addr;
    std::size_t nbytes;
  };

  static void Init(std::size_t nbytes);
  static CUDAMemPool* Get();
  
  CUDAMemPool(std::size_t nbytes);
  std::shared_ptr<PoolEntry> Alloc(std::size_t nbytes);
  std::shared_ptr<PoolEntry> Resize(std::shared_ptr<PoolEntry> entry, std::size_t nbytes);
  void CopyFromTo(std::shared_ptr<PoolEntry> src, std::shared_ptr<PoolEntry> dst);

 private:
  static std::unique_ptr<CUDAMemPool> cuda_mem_pool_;
  static bool CmpPoolEntryByAddr(const PoolEntry &a, const PoolEntry &b);

  std::shared_ptr<PoolEntry> AllocUnCheckUnlock(std::size_t nbytes);
  void Free(PoolEntry entry);
  inline size_t AlignSize(size_t nbytes) {
    return (nbytes + 1023) & ~1023;
  }

  std::mutex mutex_;
  cudaStream_t stream_;
  std::set<PoolEntry, decltype(&CUDAMemPool::CmpPoolEntryByAddr)> entry_by_addr_;
};

}
}

#endif