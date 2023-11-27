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

#include <cuda_runtime_api.h>

namespace colserve {
namespace sta {

namespace bip = boost::interprocess;

enum class MemType {
  kFree,
  kInfer,
  kTrain,
  kMemTypeNum,
};



class CUDAMemPool;
class CUDAMemPoolImpl {
public:
  struct MemPoolConfig {
    int cuda_device;
    size_t cuda_memory_size;
    std::string shared_memory_name;
    size_t shared_memory_size;
  };
  struct PoolEntry {
    void *addr;
    std::size_t nbytes;
    MemType mtype;
  };

  CUDAMemPoolImpl(MemPoolConfig config, bool force_master, bool no_cuda=false);

  ~CUDAMemPoolImpl();

  void ReleaseMempool();
  
  std::shared_ptr<PoolEntry> Alloc(std::size_t nbytes, MemType mtype);

  std::shared_ptr<PoolEntry> Resize(std::shared_ptr<PoolEntry> entry, std::size_t nbytes);

  void DumpSummary();

  void CopyFromTo(void *dst_dev_ptr, void *src_dev_ptr, size_t nbytes);

  inline size_t InferMemUsage() {
    return stat_->at(static_cast<size_t>(MemType::kInfer));
  }
  inline size_t TrainMemUsage() {
    return stat_->at(static_cast<size_t>(MemType::kTrain));
  }

  bool CheckAddr(void *addr);

  friend class CUDAMemPool;
  friend class MempoolSampler;
private:
  using Addr2EntryType = std::pair<std::ptrdiff_t, bip::managed_shared_memory::handle_t>;
  using Addr2EntryAllocator = bip::allocator<Addr2EntryType, bip::managed_shared_memory::segment_manager>;
  using Addr2Entry = boost::unordered_map<std::ptrdiff_t, bip::managed_shared_memory::handle_t, boost::hash<std::ptrdiff_t>, std::equal_to<>, Addr2EntryAllocator>;

  using Size2EntryType = std::pair<const size_t, bip::managed_shared_memory::handle_t>;
  using Size2EntryAllocator = bip::allocator<Size2EntryType, bip::managed_shared_memory::segment_manager>;
  using Size2Entry = boost::container::multimap<size_t, bip::managed_shared_memory::handle_t, std::less<>, Size2EntryAllocator>;

  using RefCount = int;
  using EntryHandle = bip::managed_shared_memory::handle_t;

  // statistic, to add more info
  using StatValueType = std::atomic<size_t>;
  using StatMap = std::array<StatValueType, static_cast<size_t>(MemType::kMemTypeNum)>;

  struct PoolEntryImpl {
    std::ptrdiff_t addr_offset;
    std::size_t nbytes;
    EntryHandle prev;
    EntryHandle next;
    MemType mtype;
    bool allocate;
  };

  bip::managed_shared_memory segment_;
  bip::interprocess_mutex *mutex_;
  Addr2Entry *addr2entry_;
  Size2Entry *size2entry_;
  RefCount *ref_count_;
  MemPoolConfig config_;
  PoolEntryImpl *empty_;
  cudaIpcMemHandle_t *cuda_mem_handle_;
  cudaStream_t cuda_memcpy_stream_;

  StatMap *stat_;

  bool master_;
  void *mem_pool_base_ptr_{nullptr};

  inline PoolEntryImpl *GetEntry(EntryHandle handle);
  inline bip::managed_shared_memory::handle_t GetHandle(PoolEntryImpl *entry);

  inline void UpdateFreeEntrySize(Size2Entry::iterator iter, PoolEntryImpl *entry, size_t newSize);
  inline void UpdateEntryAddr(const Addr2Entry::iterator &iter, std::ptrdiff_t newAddr);
  inline void ConnectPoolEntryHandle(PoolEntryImpl *eh1, PoolEntryImpl *eh2);

  inline bool CheckMemPool();

  std::shared_ptr<PoolEntry> MakeSharedPtr(PoolEntryImpl *eh, MemType mtype);

  void Free(PoolEntryImpl *entry, MemType mtype);

  void WaitSlaveExit();
};


class CUDAMemPool {
 public:
  using PoolEntry = CUDAMemPoolImpl::PoolEntry;
  static void Init(std::size_t nbytes, bool master, bool no_cuda=false);
  static CUDAMemPool* Get();
  static size_t InferMemUsage();
  static size_t TrainMemUsage();
  static void ReleaseMempool();

  static std::shared_ptr<PoolEntry> RawAlloc(size_t nbytes, MemType mtype);

  CUDAMemPool(std::size_t nbytes, bool master, bool no_cuda=false);
  ~CUDAMemPool();
  std::shared_ptr<PoolEntry> Alloc(std::size_t nbytes, MemType mtype);
  std::shared_ptr<PoolEntry> Resize(std::shared_ptr<PoolEntry> entry, std::size_t nbytes);
  void CopyFromTo(std::shared_ptr<PoolEntry> src, std::shared_ptr<PoolEntry> dst);

  inline bool CheckAddr(void *addr) {
    return impl_->CheckAddr(addr);
  }

 private:
  static std::unique_ptr<CUDAMemPool> cuda_mem_pool_;
  CUDAMemPoolImpl *impl_;
};


  extern CUDAMemPoolImpl::MemPoolConfig mempool_config_template;


}
}

#endif