#ifndef COLSERVE_CUDA_ALLOCATOR_H
#define COLSERVE_CUDA_ALLOCATOR_H

#include <common/util.h>
#include <mpool/pages_pool.h>
#include <mpool/caching_allocator.h>
#include <mpool/direct_allocator.h>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/container/map.hpp>
#include <boost/unordered_map.hpp>
#include <boost/functional/hash.hpp>
#include <boost/thread/lock_guard.hpp>

#include <functional>
#include <memory>
#include <atomic>
#include <vector>

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
    mpool::MemBlock *block;
  };
  static void Init(std::size_t nbytes, bool cleanup, bool observe, 
                   FreeListPolicyType free_list_policy);
  static CUDAMemPool* Get();
  static bool IsEnable();
  static size_t InferMemUsage(int device_id);
  static size_t TrainMemUsage(int device_id);
  static size_t TrainPeakMemUsage(int device_id);
  static size_t TrainAllMemUsage(int device_id);
  static size_t FreeMemUsage(int device_id);
  static size_t PoolNbytes(int device_id);
  static void FreeTrainLocals(int device_id);
  static void DumpDumpBlockList();

  static double TrainAllocMs();
  static void ResetTrainAllocMs();


  static std::shared_ptr<PoolEntry> RawAlloc(int device_id, size_t nbytes, MemType mtype);
  static std::shared_ptr<PoolEntry> Alloc(int device_id, std::size_t nbytes, 
                                          MemType mtype, bool allow_nullptr);

  void RegisterOOMHandler(std::function<void()> oom_handler, MemType mtype);

  CUDAMemPool(std::size_t nbytes, bool cleanup, bool observe, FreeListPolicyType free_list_policy);

  inline bool CheckAddr(void *addr) {
    // return impl_->CheckAddr(addr);
    return true;
  }

  ~CUDAMemPool();


 private:
  static bool allocate_tensor_from_memory_pool_;
  static std::unique_ptr<CUDAMemPool> cuda_mem_pool_;

  std::vector<mpool::SharableObject<mpool::PagesPool> *> pages_pools_;
  std::vector<mpool::SharableObject<mpool::CachingAllocator> *> torch_allocators_;
  std::vector<mpool::SharableObject<mpool::DirectAllocator> *> tvm_allocators_;

  // std::atomic<size_t> train_alloc_us_;
};

  // extern CUDAMemPoolImpl::MemPoolConfig mempool_config_template;

}
}

#endif