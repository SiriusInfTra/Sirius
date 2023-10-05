#ifndef COLSERVE_CUDA_ALLOCATOR_H
#define COLSERVE_CUDA_ALLOCATOR_H

#include <mutex>
#include <map>
#include <set>
#include <memory>
#include <atomic>
#include <iostream>
#include <cstddef>

namespace colserve {
namespace sta {

class CUDAMemPool {
 public:
  struct PoolEntry {
    void* addr;
    std::size_t size;
  };

  static void Init(std::size_t size);
  static CUDAMemPool* Get();
  
  CUDAMemPool(std::size_t size);
  std::shared_ptr<PoolEntry> Alloc(std::size_t size);

 private:
  static std::unique_ptr<CUDAMemPool> cuda_mem_pool_;
  static bool CmpPoolEntryByAddr(const PoolEntry &a, const PoolEntry &b);

  void Free(PoolEntry entry);

  std::mutex mutex_;
  std::set<PoolEntry, decltype(&CUDAMemPool::CmpPoolEntryByAddr)> entry_by_addr_;
};

}
}

#endif