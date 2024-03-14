#ifndef COLSERVE_CUDA_ALLOCATOR_H
#define COLSERVE_CUDA_ALLOCATOR_H

#include <functional>
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

enum class FreeListPolicyType {};
inline FreeListPolicyType getFreeListPolicy(const std::string &s) {
  return static_cast<FreeListPolicyType>(0);
}
enum class MemType {
  kFree,
  kTrainLocalFree,
  kMemTypeFreeNum,

  kInfer,
  kTrain,
  kMemTypeNum,

  // used in stat
  kTrainAll,
  kMemTypeStatNum,

  /*
   * Due to the nature of PyTorch async kernel execution, 
   * the memory released by PyTorch Tensor may not be immediately available.
   * Thus, `kTrainLocalFree` is introduced.
   * 
   *  kTrainLocalFree -- sync --> kFree <-> kInfer
   *          ^                    |
   *          |                    v
   *          \---- release ---- kTrain
   */
};

class CUDAMemPool {

public:
  struct PoolEntry {
    void *addr;
    std::size_t nbytes;
    MemType mtype;
  };
  static void Init(std::size_t nbytes, bool cleanup, bool observe, FreeListPolicyType free_list_policy);
  static CUDAMemPool* Get();
  static size_t InferMemUsage();
  static size_t TrainMemUsage();
  static size_t TrainAllMemUsage();
  static size_t FreeMemUsage();
  static size_t PoolNbytes();
  static void FreeTrainLocals();
  static void DumpDumpBlockList();

  static double TrainAllocMs() { 
    if (cuda_mem_pool_ == nullptr) return 0.0;
    return 1.0 * cuda_mem_pool_->train_alloc_us_.load(std::memory_order_relaxed) / 1000; 
  }
  static void ResetTrainAllocMs() { 
    if (cuda_mem_pool_ == nullptr) return;
    cuda_mem_pool_->train_alloc_us_.store(0, std::memory_order_relaxed); 
  }

  static std::shared_ptr<PoolEntry> RawAlloc(size_t nbytes, MemType mtype);

  void RegisterOOMHandler(std::function<void()> oom_handler, MemType mtype);

  CUDAMemPool(std::size_t nbytes, bool cleanup, bool observe, FreeListPolicyType free_list_policy);
  ~CUDAMemPool();
  std::shared_ptr<PoolEntry> Alloc(std::size_t nbytes, MemType mtype, bool allow_nullptr);
  // std::shared_ptr<PoolEntry> Resize(std::shared_ptr<PoolEntry> entry, std::size_t nbytes);
  // void CopyFromTo(std::shared_ptr<PoolEntry> src, std::shared_ptr<PoolEntry> dst);

  inline bool CheckAddr(void *addr) {
    // return impl_->CheckAddr(addr);
    return true;
  }

 private:
  static std::unique_ptr<CUDAMemPool> cuda_mem_pool_;
  MemPool *impl_;

  std::atomic<size_t> train_alloc_us_;
};


  // extern CUDAMemPoolImpl::MemPoolConfig mempool_config_template;


}
}

#endif