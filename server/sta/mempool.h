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
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <thread>
#include <utility>

#include <cuda_runtime_api.h>


#ifdef NO_CUDA

#define CUDA_TYPE(cuda_type) long

#else

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
using EntryAddrTableType = std::pair<std::ptrdiff_t, MemPoolEntryHandle>;
using EntryAddrTableAllocator = bip::allocator<EntryAddrTableType, bip::managed_shared_memory::segment_manager>;
using EntryAddrTable = boost::unordered_map<std::ptrdiff_t, bip::managed_shared_memory::handle_t, boost::hash<std::ptrdiff_t>, std::equal_to<>, EntryAddrTableAllocator>;

using EntrySizeTableType = std::pair<const size_t, MemPoolEntryHandle>;
using EntrySizeTableAllocator = bip::allocator<EntrySizeTableType, bip::managed_shared_memory::segment_manager>;
using EntrySizeTable = boost::container::multimap<size_t, MemPoolEntryHandle, std::less<>, EntrySizeTableAllocator>;
using EntrySizeTableIterator = EntrySizeTable::iterator;

using EntryListAllocator = bip::allocator<MemPoolEntryHandle, bip::managed_shared_memory::segment_manager>;
using EntryList = bip::list<MemPoolEntryHandle, EntryListAllocator>;
using EntryListIterator = EntryList::iterator;

using StatValueType = std::atomic<size_t>;
using StatMap = std::array<StatValueType, static_cast<size_t>(MemType::kMemTypeNum)>;
using RefCount = int;
struct MemPoolEntry {
  std::ptrdiff_t addr_offset;
  std::size_t nbytes;
  MemType mtype;
  EntryListIterator mem_entry_pos;
  union {
    EntryListIterator freelist_pos;
    EntrySizeTableIterator freetable_pos;
  };
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

inline MemPoolEntry *GetPrevEntry(const shared_memory &segment, MemPoolEntry *entry, const EntryListIterator &begin_bound_check) {
  auto iter = entry->mem_entry_pos;
  if (iter == begin_bound_check) {
    return nullptr;
  }
  iter--;
  return GetEntry(segment, *iter);
}

inline MemPoolEntry *GetNextEntry(const shared_memory &segment, MemPoolEntry *entry, const EntryListIterator &end_bound_check) {
  auto iter = entry->mem_entry_pos;
  iter++;
  if (iter == end_bound_check) {
    return nullptr;
  }
  return GetEntry(segment, *iter);
}

enum class FreeListPolicyType {
  kNextFit,
  kFirstFit,
  kBestFit,
  kReserved,
  kPolicyNum,
};
FreeListPolicyType getFreeListPolicy(const std::string& s);

class FreeListPolicy {
protected:
  const shared_memory &segment_;
public:
  FreeListPolicy(shared_memory &segment) : segment_(segment) {};

  virtual ~FreeListPolicy() = default;

  virtual void InitMaster(MemPoolEntry *free_entry) = 0;

  virtual void InitSlave() = 0;

  virtual MemPoolEntry *GetFreeBlock(size_t nbytes) = 0;

  virtual void NotifyUpdateFreeBlockNbytes(MemPoolEntry *entry, size_t old_nbytes) = 0;

  virtual void RemoveFreeBlock(MemPoolEntry *entry) = 0;

  virtual void AddFreeBlock(MemPoolEntry *entry) = 0;

  virtual void CheckFreeList(const EntryListIterator &begin, const EntryListIterator &end) = 0;

  virtual void DumpFreeList(std::ostream &stream, const EntryListIterator &begin, const EntryListIterator &end) = 0;
};

class BestFitPolicy: public FreeListPolicy {

private:
  EntrySizeTable *free_entry_table;
public:
  BestFitPolicy(shared_memory& segment);

  ~BestFitPolicy() {

  }

  void InitMaster(MemPoolEntry* free_entry) override;

  void InitSlave() override {}

  MemPoolEntry* GetFreeBlock(size_t nbytes) override;

  void NotifyUpdateFreeBlockNbytes(MemPoolEntry* entry,
                                  size_t old_nbytes) override;

  void RemoveFreeBlock(MemPoolEntry* entry) override;

  void AddFreeBlock(MemPoolEntry* entry) override;

  void DumpFreeList(std::ostream& stream, const EntryListIterator& begin, const EntryListIterator& end) override {
    stream << "start,len,allocated,next,prev,mtype" << std::endl;
    for (auto&& [nbytes, handle] : *free_entry_table) {
      auto* entry = GetEntry(segment_, handle);
      auto* prev = GetPrevEntry(segment_, entry, begin);
      auto* next = GetNextEntry(segment_, entry, end);
      stream << entry->addr_offset << "," << entry->nbytes << ","
            << static_cast<int>(entry->mtype) << ","
            << (prev ? next->addr_offset : -1) << ","
            << (prev ? prev->addr_offset : -1) << ","
            << static_cast<unsigned>(entry->mtype) << std::endl;
      }
  }

  void CheckFreeList(const EntryListIterator& begin, const EntryListIterator& end) override;
};



class NextFitPolicy: public FreeListPolicy {
protected:
  EntryList *freelist_;
  EntryListIterator *freelist_pos_;
public:
  NextFitPolicy(shared_memory& segment): FreeListPolicy(segment) {
    freelist_ = segment.find_or_construct<EntryList>("FreeList")(segment.get_segment_manager());
    freelist_pos_ = segment.find_or_construct<EntryListIterator>("FreeListPos")();
  }

  ~NextFitPolicy() {}

  void InitMaster(MemPoolEntry *free_entry) override {
    free_entry->freelist_pos = freelist_->insert(freelist_->end(), GetHandle(segment_, free_entry));
  }
 
  void InitSlave() override {}

  MemPoolEntry* GetFreeBlock(size_t nbytes) override {
    {
    auto iter = *freelist_pos_;
    while (iter != freelist_->cend()) {
      auto *entry = GetEntry(segment_, *iter);
      if (entry->nbytes >= nbytes) {
        *freelist_pos_ = iter;
        return entry;
      }
      iter++;
    }
  }
  {
    auto iter = freelist_->begin();
    while (iter != *freelist_pos_) {
      auto *entry = GetEntry(segment_, *iter);
      if (entry->nbytes >= nbytes) {
        *freelist_pos_ = iter;
        return entry;
      }
      iter++;
    }
  }
  return nullptr;
 }

 void NotifyUpdateFreeBlockNbytes(MemPoolEntry* entry,
                                  size_t old_nbytes) override;

 void AddFreeBlock(MemPoolEntry* entry) override;

 void RemoveFreeBlock(MemPoolEntry* entry) override;

 void CheckFreeList(const EntryListIterator& begin,
                    const EntryListIterator& end) override;

 void DumpFreeList(std::ostream& stream, const EntryListIterator& begin,
                   const EntryListIterator& end) override;
};

class FirstFitPolicy: public NextFitPolicy {
public:
  FirstFitPolicy(shared_memory &segment): NextFitPolicy(segment) {}; 

  ~FirstFitPolicy() {}

  MemPoolEntry* GetFreeBlock(size_t nbytes) override;
};

class MemPool {
  friend class MempoolSampler;
private:
  shared_memory segment_;
  FreeListPolicy *freeblock_policy_;
  EntryAddrTable *mem_entry_table_;
  EntryList *mem_entry_list_;

  MemPoolConfig config_;
  RefCount *ref_count_;
  bip::interprocess_mutex *mutex_;
  bool master_;
  bool observe_;

  std::byte *mem_pool_base_ptr_;
  CUDA_TYPE(cudaIpcMemHandle_t) *cuda_mem_handle_;
  CUDA_TYPE(cudaStream_t) cuda_memcpy_stream_;

  StatMap *stat_;

  void RemoveMemPoolEntry(MemPoolEntry *entry);

  MemPoolEntry *CreateMemPoolEntry(std::ptrdiff_t addr_offset,
                                   std::size_t nbytes, MemType mtype,
                                   EntryListIterator insert_pos);

  void Free(MemPoolEntry *entry);

  std::shared_ptr<PoolEntry> MakeSharedPtr(MemPoolEntry *entry);

  void WaitSlaveExit();

  bool CheckPoolWithoutLock();

  std::unordered_map<MemType, std::unordered_map<UsageStat, size_t>>
  GetUsageWithoutLock();

  void DumpSummaryWithoutLock();

  void CopyFromToInternel(void *dst_dev_ptr, void *src_dev_ptr, size_t nbytes);



 public:
  MemPool(MemPoolConfig config, bool cleanup, bool observe, FreeListPolicyType policy_type = FreeListPolicyType::kNextFit);

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