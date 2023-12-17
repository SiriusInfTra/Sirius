#ifndef COLSERVE_CUDA_ALLOCATOR_H
#define COLSERVE_CUDA_ALLOCATOR_H

#include <mutex>
#include <map>
#include <set>
#include <memory>
#include <atomic>
#include <array>
#include <iostream>
#include <thread>
#include <chrono>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/container/map.hpp>
#include <boost/unordered_map.hpp>
#include <boost/functional/hash.hpp>
#include <boost/thread/lock_guard.hpp>
#include "mempool.h"
#include <cuda_runtime_api.h>

namespace colserve {
namespace sta {

class CUDAMemPool {
 public:
  using PoolEntry = colserve::sta::PoolEntry;
  static void Init(std::size_t nbytes, bool cleanup, bool observe, FreeListPolicyType free_list_policy);
  static CUDAMemPool* Get();
  static size_t InferMemUsage();
  static size_t TrainMemUsage();
  static size_t FreeMemUsage();
  static size_t PoolNbytes();

  static std::shared_ptr<PoolEntry> RawAlloc(size_t nbytes, MemType mtype);

  CUDAMemPool(std::size_t nbytes, bool cleanup, bool observe, FreeListPolicyType free_list_policy);
  ~CUDAMemPool();
  std::shared_ptr<PoolEntry> Alloc(std::size_t nbytes, MemType mtype, bool allow_nullptr);
  std::shared_ptr<PoolEntry> Resize(std::shared_ptr<PoolEntry> entry, std::size_t nbytes);
  void CopyFromTo(std::shared_ptr<PoolEntry> src, std::shared_ptr<PoolEntry> dst);

  inline bool CheckAddr(void *addr) {
    // return impl_->CheckAddr(addr);
    return true;
  }

 private:
  static std::unique_ptr<CUDAMemPool> cuda_mem_pool_;
  MemPool *impl_;
};


  // extern CUDAMemPoolImpl::MemPoolConfig mempool_config_template;


}
}

#endif