#ifndef COLSERVE_CUDA_MEMPOOL_H
#define COLSERVE_CUDA_MEMPOOL_H
#include <algorithm>
#include <atomic>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/list.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/container/map.hpp>
#include <boost/unordered_map.hpp>
#include <boost/functional/hash.hpp>
#include <boost/thread/lock_guard.hpp>
#include <cassert>
#include <cstddef>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <thread>
#include <utility>




#ifdef NO_CUDA

#define CUDA_CALL0(func)
#define CUDA_TYPE(cuda_type) long

#else

#include <cuda_runtime_api.h>
#define CUDA_CALL0(func) do { \
  auto error = func; \
  if (error != cudaSuccess) { \
    LOG(FATAL) << #func << " " << cudaGetErrorString(error); \
    exit(EXIT_FAILURE); \
  } \
  } while (0)

#define CUDA_TYPE(cuda_type) cuda_type 

#endif


namespace colserve {
namespace sta {
namespace bip = boost::interprocess;

enum class MemType {
  kFree,
  kInfer,
  kTrain,
  kMemTypeNum,
};

enum class UsageStat {
  kMaxNBytes,
  kMinNBytes,
  kCount,
  kTotalNBytes,
};


namespace detail {
inline size_t GetAlignedNbytes(size_t nbytes) {
  constexpr size_t alignment = 1024;
  static_assert((alignment & (alignment - 1)) == 0, "alignment must be power of 2");
  return (nbytes + (alignment - 1)) & (~(alignment - 1));
}
inline double ByteToMB(size_t nbytes) {
  return static_cast<double>(nbytes) / 1024 / 1024;
}

inline std::string ByteDisplay(size_t nbytes) {
  std::stringstream ss;
  ss << static_cast<double>(nbytes) / 1024 / 1024 << "MB(" << nbytes << "bytes)";
  return ss.str();
}
}

using shared_memory = bip::managed_shared_memory;
using segment_manager = shared_memory::segment_manager;

using MemPoolEntryHandle = bip::managed_shared_memory::handle_t;
using MemEntryTableType = std::pair<std::ptrdiff_t, MemPoolEntryHandle>;
using MemEntryTableAllocator = bip::allocator<MemEntryTableType, bip::managed_shared_memory::segment_manager>;
using MemEntryTable = boost::unordered_map<std::ptrdiff_t, bip::managed_shared_memory::handle_t, boost::hash<std::ptrdiff_t>, std::equal_to<>, MemEntryTableAllocator>;

using MemEntryListAllocator = bip::allocator<MemPoolEntryHandle, bip::managed_shared_memory::segment_manager>;
using MemEntryList = bip::list<MemPoolEntryHandle, MemEntryListAllocator>;
using MemEntryListIterator = MemEntryList::iterator;

using StatValueType = std::atomic<size_t>;
using StatMap = std::array<StatValueType, static_cast<size_t>(MemType::kMemTypeNum)>;
using RefCount = int;
struct MemPoolEntry {
  std::ptrdiff_t addr_offset;
  std::size_t nbytes;
  MemType mtype;
  MemEntryListIterator mem_entry_pos;
  MemEntryListIterator freelist_pos;
};

inline std::ostream &operator<<(std::ostream &os, const MemPoolEntry &entry) {
  os << "entry: {addr_offset=" << entry.addr_offset << ", nbytes=" << entry.nbytes
    << ", mtype=" << static_cast<int>(entry.mtype) << "}";
  return os;
}

struct PoolEntry {
  void *addr;
  std::size_t nbytes;
  MemType mtype;
};

struct MemPoolConfig {
  int cuda_device;
  size_t cuda_memory_size;
  std::string shared_memory_name;
  size_t shared_memory_size;
};

static inline MemPoolConfig GetDefaultMemPoolConfig(size_t nbytes) {
  return {
    .cuda_device = 0,
    .cuda_memory_size = nbytes,
    .shared_memory_name = "gpu_colocation_mempool",
    .shared_memory_size = 1024 * 1024 * 1024, /* 1G */
  };
}

inline MemPoolEntry *GetEntry(const shared_memory &segment, MemPoolEntryHandle handle) {
  return reinterpret_cast<MemPoolEntry *>(segment.get_address_from_handle(handle));
}

inline MemPoolEntryHandle GetHandle(const shared_memory &segment, MemPoolEntry *entry) {
  return segment.get_handle_from_address(entry);
}

class FirstFitPolicy {
  friend class MempoolSampler;
private:
  const shared_memory &segment_;
  MemEntryList *freelist_;
  MemEntryListIterator *freelist_pos_;
public:
  FirstFitPolicy(shared_memory &segment);

  void InitMaster(MemPoolEntry *free_entry);

  void InitSlave();

  MemPoolEntry *GetFreeBlock(size_t nbytes);

  void NotifyUpdateFreeBlockNbytes(MemPoolEntry *entry, size_t old_nbytes);

  void RemoveFreeBlock(MemPoolEntry *entry);

  void AddFreeBlock(MemPoolEntry *entry);

  void CheckFreeList(const MemEntryListIterator &begin,
                     const MemEntryListIterator &end);
};


class MemPool {
  friend class MempoolSampler;
private:
  shared_memory segment_;
  FirstFitPolicy *freeblock_policy_;
  MemEntryTable *mem_entry_table_;
  MemEntryList *mem_entry_list_;

  MemPoolConfig config_;
  RefCount *ref_count_;
  bip::interprocess_mutex *mutex_;
  bool master_;
  bool observe_;

  std::byte *mem_pool_base_ptr_;
  CUDA_TYPE(cudaIpcMemHandle_t) *cuda_mem_handle_;
  CUDA_TYPE(cudaStream_t) cuda_memcpy_stream_;

  StatMap *stat_;

  MemPoolEntry *GetPrevEntry(MemPoolEntry *entry);

  MemPoolEntry *GetNextEntry(MemPoolEntry *entry);

  void RemoveMemPoolEntry(MemPoolEntry *entry);

  MemPoolEntry *CreateMemPoolEntry(std::ptrdiff_t addr_offset,
                                   std::size_t nbytes, MemType mtype,
                                   MemEntryListIterator insert_pos);

  void Free(MemPoolEntry *entry);

  std::shared_ptr<PoolEntry> MakeSharedPtr(MemPoolEntry *entry);

  void WaitSlaveExit();

  bool CheckPoolWithoutLock();

  std::unordered_map<MemType, std::unordered_map<UsageStat, size_t>>
  GetUsageWithoutLock();

  void DumpSummaryWithoutLock();

  void CopyFromToInternel(void *dst_dev_ptr, void *src_dev_ptr, size_t nbytes);



 public:
  MemPool(MemPoolConfig config, bool cleanup, bool observe);

  ~MemPool();

  void CheckPool();


  std::shared_ptr<PoolEntry> Alloc(std::size_t nbytes, MemType mtype);

  void CopyFromTo(std::shared_ptr<PoolEntry> src, std::shared_ptr<PoolEntry> dst);

  inline size_t InferMemUsage() {
    return stat_->at(static_cast<size_t>(MemType::kInfer)).load(std::memory_order_relaxed);
  }
  inline size_t TrainMemUsage() {
    return stat_->at(static_cast<size_t>(MemType::kTrain)).load(std::memory_order_relaxed);
  }

  inline size_t PoolNbytes() {
    return config_.cuda_memory_size;
  }

  inline void DumpSummary() {
    bip::scoped_lock locker(*mutex_);
    DumpSummaryWithoutLock();
  }

  inline std::unordered_map<MemType, std::unordered_map<UsageStat, size_t>> GetUsage() {
    bip::scoped_lock locker(*mutex_);
    return GetUsageWithoutLock();
  }
};


}
}



#endif