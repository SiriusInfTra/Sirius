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


#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/container/map.hpp>
#include <boost/unordered_map.hpp>
#include <boost/functional/hash.hpp>
#include <boost/thread/lock_guard.hpp>


#include <cuda_runtime_api.h>

#include <thread>
#include <chrono>
#include <iostream>


namespace colserve {
namespace sta {

namespace bip = boost::interprocess;

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
  };

  explicit CUDAMemPoolImpl(MemPoolConfig config, bool force_master);

  ~CUDAMemPoolImpl();

  std::shared_ptr<PoolEntry> Alloc(std::size_t nbytes);

  std::shared_ptr<PoolEntry> Resize(std::shared_ptr<PoolEntry> entry, std::size_t nbytes);

  void DumpSummary();

  void CopyFromTo(void *dst_dev_ptr, void *src_dev_ptr, size_t nbytes);

private:
  using Addr2EntryType = std::pair<std::ptrdiff_t, bip::managed_shared_memory::handle_t>;
  using Addr2EntryAllocator = bip::allocator<Addr2EntryType, bip::managed_shared_memory::segment_manager>;
  using Addr2Entry = boost::unordered_map<std::ptrdiff_t, bip::managed_shared_memory::handle_t, boost::hash<std::ptrdiff_t>, std::equal_to<>, Addr2EntryAllocator>;

  using Size2EntryType = std::pair<const size_t, bip::managed_shared_memory::handle_t>;
  using Size2EntryAllocator = bip::allocator<Size2EntryType, bip::managed_shared_memory::segment_manager>;
  using Size2Entry = boost::container::multimap<size_t, bip::managed_shared_memory::handle_t, std::less<>, Size2EntryAllocator>;

  using RefCount = int;
  using EntryHandle = bip::managed_shared_memory::handle_t;

  struct PoolEntryImpl {
    std::ptrdiff_t addr_offset;
    std::size_t nbytes;
    EntryHandle prev;
    EntryHandle next;
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

  bool master_;
  void *devPtr_{};

  inline PoolEntryImpl *GetEntry(EntryHandle handle);

  inline bip::managed_shared_memory::handle_t GetHandle(PoolEntryImpl *entry);

  inline void UpdateFreeEntrySize(Size2Entry::iterator iter, PoolEntryImpl *entry, size_t newSize);

  inline void UpdateEntryAddr(const Addr2Entry::iterator &iter, std::ptrdiff_t newAddr);

  inline void ConnectPoolEntryHandle(PoolEntryImpl *eh1, PoolEntryImpl *eh2);

  inline void CheckMemPool();

  std::shared_ptr<PoolEntry> MakeSharedPtr(PoolEntryImpl *eh);

  void Free(PoolEntryImpl *entry);

};


class CUDAMemPool {
 public:
  using PoolEntry = CUDAMemPoolImpl::PoolEntry;
  static void Init(std::size_t nbytes, bool master);
  static CUDAMemPool* Get();

  static std::shared_ptr<PoolEntry> RawAlloc(size_t nbytes);

  CUDAMemPool(std::size_t nbytes, bool master);
  std::shared_ptr<PoolEntry> Alloc(std::size_t nbytes);
  std::shared_ptr<PoolEntry> Resize(std::shared_ptr<PoolEntry> entry, std::size_t nbytes);
  void CopyFromTo(std::shared_ptr<PoolEntry> src, std::shared_ptr<PoolEntry> dst);

 private:
  static std::unique_ptr<CUDAMemPool> cuda_mem_pool_;
  CUDAMemPoolImpl *impl_;
};



}
}

#endif