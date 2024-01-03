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
#include <iostream>
#include <iomanip>

#include <cuda_runtime_api.h>


#ifdef NO_CUDA

#define CUDA_TYPE(cuda_type) long

#else

#define CUDA_TYPE(cuda_type) cuda_type 

#endif


namespace colserve {
namespace sta {
namespace detail {
constexpr size_t alignment = 1024;
constexpr size_t train_alloc_threshold = 256 * 1024 * 1024;
constexpr size_t train_alloc_threshold_small = 32 * 1024 * 1024;
inline size_t GetAlignedNbytes(size_t nbytes) {
  static_assert((alignment & (alignment - 1)) == 0, "alignment must be power of 2");
  return (nbytes + (alignment - 1)) & (~(alignment - 1));
}
inline size_t GetAlignedNbytes(size_t nbytes, size_t alignment_) {
  assert((alignment_ & (alignment_ - 1)) == 0);
  return (nbytes + (alignment_ - 1)) & (~(alignment_ - 1));
}
inline double ByteToMB(size_t nbytes) {
  return static_cast<double>(nbytes) / 1024 / 1024;
}

inline std::string ByteDisplay(size_t nbytes) {
  std::stringstream ss;
  ss << std::fixed << std::setprecision(2)
     << static_cast<double>(nbytes) / 1024 / 1024 << "MB (" << nbytes << " Bytes)";
  return ss.str();
}
} // namespace detail

namespace bip = boost::interprocess;
using bip_shared_memory = bip::managed_shared_memory;
using bip_segment_manager = bip_shared_memory::segment_manager;

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

std::ostream &operator<<(std::ostream &os, const MemType &mtype);

enum class UsageStat {
  kMaxNBytes,
  kMinNBytes,
  kCount,
  kTotalNBytes,
};

struct MemPoolEntry {
  std::ptrdiff_t addr_offset;
  std::size_t nbytes;
  MemType mtype;
  EntryListIterator mem_entry_pos;
  union {
    EntryListIterator freelist_pos;
    EntrySizeTableIterator freetable_pos;
  };

  inline bool IsMergableFree(MemType cur_mtype) const {
    return cur_mtype == MemType::kTrain ? mtype == MemType::kTrainLocalFree : mtype == MemType::kFree;
  }
  inline bool IsAvailableFree(MemType cur_mtype) const {
    return (mtype == MemType::kFree) || (cur_mtype == MemType::kTrain && mtype == MemType::kTrainLocalFree);
  }
  inline bool IsFree() const {
    return (mtype == MemType::kFree) || (mtype == MemType::kTrainLocalFree);
  }
  inline void SetAsFree() {
    mtype = mtype == MemType::kTrain ? MemType::kTrainLocalFree : MemType::kFree;
  }
};

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

using StatValueType = std::atomic<size_t>;
using StatMap = std::array<StatValueType, static_cast<size_t>(MemType::kMemTypeStatNum)>;
using RefCount = int;

inline std::ostream &operator<<(std::ostream &os, const MemPoolEntry &entry) {
  os << "entry: {addr_offset=" << entry.addr_offset << ", nbytes=" << entry.nbytes
     << ", mtype=" << static_cast<int>(entry.mtype) << "}";
  return os;
}

inline MemPoolEntry *GetEntry(const bip_shared_memory &segment, MemPoolEntryHandle handle) {
  return reinterpret_cast<MemPoolEntry *>(segment.get_address_from_handle(handle));
}

inline MemPoolEntryHandle GetHandle(const bip_shared_memory &segment, MemPoolEntry *entry) {
  return segment.get_handle_from_address(entry);
}

inline MemPoolEntry *GetPrevEntry(const bip_shared_memory &segment, MemPoolEntry *entry, 
                                  const EntryListIterator &begin_bound_check) {
  auto iter = entry->mem_entry_pos;
  if (iter == begin_bound_check) {
    return nullptr;
  }
  iter--;
  return GetEntry(segment, *iter);
}

inline MemPoolEntry *GetNextEntry(const bip_shared_memory &segment, MemPoolEntry *entry, 
                                  const EntryListIterator &end_bound_check) {
  auto iter = entry->mem_entry_pos;
  iter++;
  if (iter == end_bound_check) {
    return nullptr;
  }
  return GetEntry(segment, *iter);
}


static inline MemPoolConfig GetDefaultMemPoolConfig(size_t nbytes) {
  return {
    .cuda_device = 0,
    .cuda_memory_size = nbytes,
    .shared_memory_name = "gpu_colocation_mempool",
    .shared_memory_size = 1024 * 1024 * 1024, /* 1G */
  };
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
  using PolicyName = std::array<char, 20>;

  const bip_shared_memory &segment_;
  PolicyName *policy_name_;
  std::string policy_name_str_{"FreeListPolicy"};

  virtual bool CheckGetFreeInput(size_t nbytes, MemType mtype) {
    return mtype == MemType::kInfer || mtype == MemType::kTrain;
  }
public:
  FreeListPolicy(bip_shared_memory &segment) : segment_(segment) {
    policy_name_ = segment.find_or_construct<PolicyName>("PolicyName")();
  };

  virtual ~FreeListPolicy() = default;

  virtual void InitMaster(MemPoolEntry *free_entry);
  virtual void InitSlave();

  virtual MemPoolEntry *GetFreeBlock(size_t nbytes, MemType mtype, bool local, bool global) = 0;
  virtual MemPoolEntry* GetFreeBlockByMerge(size_t nbytes, MemType mtype, EntryList* mem_entry_list) {
    return nullptr;
  }

  virtual void NotifyUpdateFreeBlockNbytes(MemPoolEntry *entry, size_t old_nbytes) = 0;

  virtual void RemoveFreeBlock(MemPoolEntry *entry) = 0;

  virtual void AddFreeBlock(MemPoolEntry *entry) = 0;

  virtual void RemoveLocalFreeBlocks(MemType mtype, std::function<void(MemPoolEntry*)> fn) = 0;

  virtual void CheckFreeList(const EntryListIterator &begin, const EntryListIterator &end) = 0;

  virtual void DumpFreeList(std::ostream &stream, const EntryListIterator &begin, const EntryListIterator &end) = 0;
};


class BestFitPolicy : public FreeListPolicy {
 private:
  EntrySizeTable *free_entry_tables_[static_cast<size_t>(MemType::kMemTypeFreeNum)];
 public:
  BestFitPolicy(bip_shared_memory& segment);

  ~BestFitPolicy() {}

  void InitMaster(MemPoolEntry* free_entry) override;

  MemPoolEntry* GetFreeBlock(size_t nbytes, MemType mtype, bool local, bool global) override;
  MemPoolEntry* GetFreeBlockByMerge(size_t nbytes, MemType mtype, EntryList* mem_entry_list) override;

  void NotifyUpdateFreeBlockNbytes(MemPoolEntry* entry,
                                   size_t old_nbytes) override;

  void RemoveFreeBlock(MemPoolEntry* entry) override;

  void AddFreeBlock(MemPoolEntry* entry) override;

  void RemoveLocalFreeBlocks(MemType mtype, std::function<void(MemPoolEntry*)> fn) override;

  void DumpFreeList(std::ostream& stream, const EntryListIterator& begin, const EntryListIterator& end) override {
    stream << "start, len, allocated, next, prev, mtype" << std::endl;
    for (size_t i = 0; i < static_cast<size_t>(MemType::kMemTypeFreeNum); i++) {
      auto free_entry_table = free_entry_tables_[i];
      for (auto&& [nbytes, handle] : *free_entry_table) {
        auto* entry = GetEntry(segment_, handle);
        auto* prev = GetPrevEntry(segment_, entry, begin);
        auto* next = GetNextEntry(segment_, entry, end);
        stream << entry->addr_offset << "," << entry->nbytes << ", "
              << static_cast<int>(entry->mtype) << ", "
              << (next ? next->addr_offset : -1) << ", "
              << (prev ? prev->addr_offset : -1) << ", "
              << static_cast<unsigned>(entry->mtype) << std::endl;
      }
    }
  }

  void CheckFreeList(const EntryListIterator& begin, const EntryListIterator& end) override;
};



class NextFitPolicy: public FreeListPolicy {
 protected:
  EntryList *freelists_[static_cast<size_t>(MemType::kMemTypeFreeNum)];
  EntryListIterator *freelist_poses_[static_cast<size_t>(MemType::kMemTypeFreeNum)];
 public:
  NextFitPolicy(bip_shared_memory& segment): FreeListPolicy(segment) {
    policy_name_str_ = "NextFitPolicy";
    for (size_t i = 0; i < static_cast<size_t>(MemType::kMemTypeFreeNum); i++) {
      std::string free_list_name = "FreeList" + std::to_string(i);
      std::string free_list_pos_name = "FreeListPos" + std::to_string(i);
      freelists_[i] = segment.find_or_construct<EntryList>(free_list_name.c_str())(segment.get_segment_manager());
      freelist_poses_[i] = segment.find_or_construct<EntryListIterator>(free_list_pos_name.c_str())();
    }
  }
  ~NextFitPolicy() {}

  void InitMaster(MemPoolEntry *free_entry) override;
 
  MemPoolEntry* GetFreeBlock(size_t nbytes, MemType mtype, bool local, bool global) override {
    auto find_free_block = [this, nbytes](
        EntryList* freelist, EntryListIterator* freelist_pos) -> MemPoolEntry* {
      {
        auto iter = *freelist_pos;
        while (iter != freelist->cend()) {
          auto *entry = GetEntry(this->segment_, *iter);
          if (entry->nbytes >= nbytes) {
            *freelist_pos = iter;
            return entry;
          }
          iter++;
        }
      }
      {
        auto iter = freelist->begin();
        while (iter != *freelist_pos) {
          auto *entry = GetEntry(this->segment_, *iter);
          if (entry->nbytes >= nbytes) {
            *freelist_pos = iter;
            return entry;
          }
          iter++;
        }
      }
      return nullptr;
    };
    if (mtype == MemType::kTrain && local) {
      auto entry = find_free_block(
          freelists_[static_cast<size_t>(MemType::kTrainLocalFree)],
          freelist_poses_[static_cast<size_t>(MemType::kTrainLocalFree)]);
      if (entry != nullptr) return entry;
    }
    if (global) {
      return find_free_block(
          freelists_[static_cast<size_t>(MemType::kFree)],
          freelist_poses_[static_cast<size_t>(MemType::kFree)]);
    } else {
      return nullptr;
    }
  }

  void NotifyUpdateFreeBlockNbytes(MemPoolEntry* entry,
                                   size_t old_nbytes) override;

  void AddFreeBlock(MemPoolEntry* entry) override;

  void RemoveFreeBlock(MemPoolEntry* entry) override;

  void RemoveLocalFreeBlocks(MemType mtype, std::function<void(MemPoolEntry*)> fn) override {
    // TODO
  };


  void CheckFreeList(const EntryListIterator& begin,
                     const EntryListIterator& end) override;

 void DumpFreeList(std::ostream& stream, const EntryListIterator& begin,
                     const EntryListIterator& end) override;
};

class FirstFitPolicy: public NextFitPolicy {
public:
  FirstFitPolicy(bip_shared_memory &segment): NextFitPolicy(segment) {
    policy_name_str_ = "FirstFitPolicy";
  }; 

  ~FirstFitPolicy() {}

  MemPoolEntry* GetFreeBlock(size_t nbytes, MemType mtype, bool local, bool global) override;
};

class MemPool {
  friend class MempoolSampler;
private:
  bip_shared_memory segment_;
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
  MemPoolEntry* FreeWithoutLock(MemPoolEntry *entry);

  std::shared_ptr<PoolEntry> MakeSharedPtr(MemPoolEntry *entry);

  void WaitSlaveExit();

  bool CheckPoolWithoutLock();

  std::unordered_map<MemType, std::unordered_map<UsageStat, size_t>>
  GetUsageWithoutLock();

  void DumpSummaryWithoutLock();
  void DumpBlockListWithoutLock();

  void CopyFromToInternel(void *dst_dev_ptr, void *src_dev_ptr, size_t nbytes);


 public:
  MemPool(MemPoolConfig config, bool cleanup, bool observe, FreeListPolicyType policy_type = FreeListPolicyType::kNextFit);

  ~MemPool();

  void CheckPool();

  std::shared_ptr<PoolEntry> Alloc(std::size_t nbytes, MemType mtype);
  void FreeLocals(MemType mtype);

  void CopyFromTo(std::shared_ptr<PoolEntry> src, std::shared_ptr<PoolEntry> dst);

  inline size_t GetMemUsage(MemType mtype) {
    return stat_->at(static_cast<size_t>(mtype)).load(std::memory_order_relaxed);
  }

  inline size_t PoolNbytes() {
    return config_.cuda_memory_size;
  }

  inline void DumpSummary() {
    bip::scoped_lock locker(*mutex_);
    DumpSummaryWithoutLock();
  }

  inline void DumpBlockList() {
    bip::scoped_lock locker(*mutex_);
    DumpBlockListWithoutLock();
  }

  inline std::unordered_map<MemType, std::unordered_map<UsageStat, size_t>> GetUsage() {
    bip::scoped_lock locker(*mutex_);
    return GetUsageWithoutLock();
  }
};


}
}



#endif