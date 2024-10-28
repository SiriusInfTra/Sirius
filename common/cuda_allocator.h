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

/**
 *  @brief CUDAMemPool is a wrapper of device memory pool.
 *      It provides a unified interface for training and inference
 *      memory allocation.
 */

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

    mpool::MemBlock *block{nullptr};
    void* extra_data{nullptr};
  };
  static void Init(int device_id, std::size_t nbytes, 
                   bool cleanup, bool observe, 
                   FreeListPolicyType free_list_policy,
                   bool enable_mpool);
  static CUDAMemPool* Get(int device_id);
  static bool IsEnable();

  static std::shared_ptr<PoolEntry> 
      HostAlloc(size_t nbytes, MemType mtype);

  CUDAMemPool(int device_id, std::size_t nbytes, 
              bool cleanup, bool observe, 
              FreeListPolicyType free_list_policy);

  size_t InferMemUsage();
  size_t TrainMemUsage();
  size_t TrainPeakMemUsage();
  size_t TrainAllMemUsage();
  size_t FreeMemUsage();
  size_t PoolNbytes();
  void FreeTrainLocals();
  void DumpDumpBlockList();
  std::shared_ptr<PoolEntry> RawAlloc(size_t nbytes, MemType mtype);

  // Alloc memory and ignore the stream property.
  std::shared_ptr<PoolEntry> Alloc(size_t nbytes, MemType mtype,
                                   bool allow_nullptr);
  std::shared_ptr<PoolEntry> AllocWithStream(std::size_t nbytes, MemType mtype, 
                                             cudaStream_t stream, bool allow_nullptr);

  void RegisterOOMHandler(std::function<void()> oom_handler, MemType mtype);

  static double TrainAllocMs();
  static void ResetTrainAllocMs();

  inline bool CheckAddr(void *addr) {
    // return impl_->CheckAddr(addr);
    return true;
  }

  ~CUDAMemPool();

 private:
  static bool allocate_tensor_from_memory_pool_;
  static std::array<std::unique_ptr<CUDAMemPool>, MAX_DEVICE_NUM> cuda_mem_pools_;

  // currently, we only allow one thread to allocate memory from
  // one device memory pool, the allocation performance is fast enough.
  std::mutex mut_; 

  int device_id_;
  int raw_alloc_enable_unified_memory_{-1};
  mpool::SharableObject<mpool::PagesPool>* pages_pool_;
  mpool::SharableObject<mpool::CachingAllocator>* torch_allocator_;
  mpool::SharableObject<mpool::DirectAllocator>* tvm_allocator_;

  bool enable_mpool_;

  // std::atomic<size_t> train_alloc_us_;
};

} // namespace sta
} // namespace colserve

#endif