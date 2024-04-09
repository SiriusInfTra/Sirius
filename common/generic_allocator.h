#pragma once

#include <common/log_as_glog_sta.h>
#include <common/mempool.h>
#include <common/util.h>

#include <boost/unordered/unordered_map.hpp>
#include <boost/container/map.hpp>
#include <boost/interprocess/containers/list.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iterator>
#include <list>
#include <map>
#include <numeric>
#include <ostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <functional>
#include <fstream>
#include <utility>

namespace colserve::sta {
static const constexpr size_t VA_RESERVE_SCALE = 16;

namespace alloc_conf {

static const constexpr bool ALWAYS_CHECK_STATE = false;
static const constexpr bool STRICT_CHECK_STATE = false;
static const constexpr bool DUMP_BEFORE_CHECK = false;
static const constexpr bool VERBOSE = false;
// static const constexpr bool VERBOSE = true;
static const constexpr bool DELAY_UNMAP = true;

}



struct MemEntry;

template<typename T>
class shm_handle {
private:
  bip::managed_shared_memory::handle_t handle_;
public:
  shm_handle(T *t): handle_(MemPool::Get().shared_memory_.get_handle_from_address(t)) {}
  T *ptr() const { return reinterpret_cast<T*>(MemPool::Get().shared_memory_.get_address_from_handle(handle_)); }

};

namespace bip = boost::interprocess;
using entry_linklist = bip::list<
    shm_handle<MemEntry>, 
    bip::allocator<shm_handle<MemEntry>, 
    bip::managed_shared_memory::segment_manager>>;

using entry_addr_map = boost::container::map<
    std::ptrdiff_t, 
    shm_handle<MemEntry>, 
    std::less<>, 
    bip::allocator<std::pair<const std::ptrdiff_t, shm_handle<MemEntry>>, 
    bip::managed_shared_memory::segment_manager>>;
                                             
using entry_nbytes_map = boost::container::multimap<
    size_t, 
    shm_handle<MemEntry>, 
    std::less<>, 
    bip::allocator<std::pair<const size_t, shm_handle<MemEntry>>, 
    bip::managed_shared_memory::segment_manager>>;

struct MemEntry {
  std::ptrdiff_t  addr_offset;
  std::size_t     nbytes;
  struct {
    bool            is_free:  1;
    bool            is_train: 1;
    bool            is_small: 1;
    bool            is_alloc: 1;
    size_t          rank: 8;
  };

  entry_linklist::iterator    pos_entrylist;
  entry_addr_map::iterator    pos_entrytable;
  entry_nbytes_map::iterator  pos_freelist;
};

inline std::pair<size_t, size_t> GetAssociatedPhyMemIndex(const MemEntry *entry) {
  size_t index_begin = entry->addr_offset / MEM_BLOCK_NBYTES;
  size_t index_end = (entry->addr_offset + entry->nbytes + MEM_BLOCK_NBYTES - 1) / MEM_BLOCK_NBYTES;
  return std::make_pair(index_begin, index_end);
}

inline std::string ToString(const MemEntry *entry) {
  if (entry == nullptr) { return "nullptr"; }
  std::stringstream ss;
  ss << "{addr_offset=" << entry->addr_offset 
    << ", nbytes=" << entry->nbytes
    << ", is_free=" << entry->is_free
    << ", is_train=" << entry->is_train
    << ", is_small=" << entry->is_small 
    << ", is_alloc=" << entry->is_alloc
    << ", rank=" << entry->rank << "}";
  return ss.str();
}


inline std::ostream & operator<<(std::ostream &os, const MemEntry *entry)  {
  os << ToString(entry);
  return os;
}



class EntryList {
private:
  const std::string &log_prefix_;
  const std::vector<PhyMem *> &mapped_mem_list_;
  entry_linklist *entry_list_;
  entry_addr_map *entry_by_addr;
public:
  EntryList(const std::string &log_prefix, const std::vector<PhyMem *> &mapped_mem_list, Belong policy): log_prefix_(log_prefix), mapped_mem_list_(mapped_mem_list) {
    auto &shared_memory = MemPool::Get().GetSharedMemory();
    auto atomic_init = [&] {
      {
        std::string name = "EL_entry_list_" + std::to_string(static_cast<size_t>(policy));
        entry_list_ = shared_memory.find_or_construct<entry_linklist>(name.c_str())(shared_memory.get_segment_manager());
      }
      {
        std::string name = "EL_entry_by_addr_" + std::to_string(static_cast<size_t>(policy));
        entry_by_addr = shared_memory.find_or_construct<entry_addr_map>(name.c_str())(shared_memory.get_segment_manager());
      }

    };
    shared_memory.atomic_func(atomic_init);
  }
  
  void LinkNewEntry(MemEntry *entry);

  void UpdateAllocFlag(MemEntry *entry) {
    bool real_allocated = true;
    auto &&[index_begin, index_end] = GetAssociatedPhyMemIndex(entry);
    for (size_t k = index_begin; k < index_end; ++k) {
      if (mapped_mem_list_[k] == nullptr) {
        real_allocated = false;
        break;
      }
    }
    entry->is_alloc = real_allocated;
  }

  MemEntry *GetEntry(std::ptrdiff_t addr_offset);

  MemEntry *GetEntryLower(std::ptrdiff_t addr_offset) {
    auto iter = entry_by_addr->lower_bound(addr_offset);
    if (iter == entry_by_addr->cend()) {
      return nullptr;
    }
    auto *entry = iter->second.ptr();
    CHECK(iter == entry->pos_entrytable); 
    return entry;
  }

  MemEntry *GetEntryUpper(std::ptrdiff_t addr_offset) {
    auto iter = entry_by_addr->upper_bound(addr_offset);
    if (iter == entry_by_addr->cend()) {
      return nullptr;
    }
    auto *entry = iter->second.ptr();
    CHECK(iter == entry->pos_entrytable); 
    return entry;
  }

  MemEntry* IterateRange(ptrdiff_t addr_offset, size_t nbytes, std::function<MemEntry*(MemEntry *)> func, bool do_check = true) {
        auto *entry = GetEntryLower(addr_offset);
    if (entry == nullptr) {
      entry = std::prev(entry_list_->cend())->ptr();
    } else if (entry->addr_offset > addr_offset) {
      entry = GetPrevEntry(entry);
    }
    CHECK(entry != nullptr);
    CHECK_LE(entry->addr_offset, addr_offset);
    while (true) {
      entry = func(entry);
      if (!entry) { break; }
      if (entry->addr_offset + entry->nbytes >= addr_offset + nbytes) { break; }
      entry = GetNextEntry(entry);
      if (!entry) { break; }
    }
    return entry;
  }


  MemEntry *GetPrevEntry(MemEntry *entry);

  MemEntry *GetNextEntry(MemEntry *entry);

  MemEntry *SplitEntry(MemEntry *origin_entry, size_t remain);

  MemEntry *MergeMemEntry(MemEntry *first_entry,
                                MemEntry *secound_entry);

  void DumpMemEntryList(std::ostream &out);

  bool CheckState();
};

class FreeList {
private:
  const std::string &log_prefix_;
  EntryList& list_index_;
  const bool is_small_;
  const Belong policy_;
  const size_t small_block_nbytes_;
  entry_nbytes_map *entry_by_nbytes_;
public:


  FreeList(EntryList &list_index, bool is_small, const std::string &log_prefix, Belong policy, size_t small_block_nbytes);

  MemEntry *PopFreeEntry(size_t nbytes, bool do_split = true, size_t require_allocated = 0);

  MemEntry *PopFreeEntry(MemEntry *free_entry);

  MemEntry *PopFreeEntryLarge(size_t nbytes);

  MemEntry* PushFreeEntry(MemEntry *entry);

  void DumpFreeList(std::ostream &out);

  bool CheckState();
};



class GenericAllocator {
public:
  const Belong policy_;
  const size_t small_block_nbytes_;
  const std::string log_prefix_;
protected:
  MemPool &mempool_;
  std::vector<PhyMem *> mapped_mem_list_;
  std::byte *base_ptr_;

  EntryList entry_list_;
  FreeList  free_list_small_;
  FreeList  free_list_large_;
  std::function<void()>       oom_handler_;


  void ExpandMemorySpace(const std::vector<PhyMem *> &phy_mem_list, size_t len) {
    if( (mapped_mem_list_.size() + len) * MEM_BLOCK_NBYTES > mempool_.mempool_nbytes * VA_RESERVE_SCALE) {
      DumpState();
      LOG(FATAL) << log_prefix_ << "VA OMM";
    }

    auto *mapping_begin = base_ptr_ + mapped_mem_list_.size() * MEM_BLOCK_NBYTES;
    for (size_t k = 0; k < len; ++k) {
      auto *mem_ptr = phy_mem_list[k];
      CU_CALL(cuMemMap(
          reinterpret_cast<CUdeviceptr>(base_ptr_ + mapped_mem_list_.size() * MEM_BLOCK_NBYTES),
          MEM_BLOCK_NBYTES, 0, mem_ptr->cu_handle, 0));
      mapped_mem_list_.push_back(mem_ptr);
    }
    CUmemAccessDesc acc_desc = {
        .location = {.type = CU_MEM_LOCATION_TYPE_DEVICE, .id = 0},
        .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    };
    CU_CALL(cuMemSetAccess(reinterpret_cast<CUdeviceptr>(mapping_begin),
                           len * MEM_BLOCK_NBYTES, &acc_desc, 1));
    // cached_nbytes_ += len * MEM_BLOCK_NBYTES;
  }
  

public:

  void DumpState() {
    if (oom_handler_ != nullptr) {
      oom_handler_();
    }
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << "./" << "mempool_dump_log" << "/"
        << "mempool_dump_" << policy_ << "_" 
        << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%H-%M-%S") << "_" << ms.count();
    std::filesystem::path output_dir{ss.str()};
    CHECK(std::filesystem::create_directories(output_dir));
    LOG(INFO) << log_prefix_ << "Dump has been written to: " << ss.str() << ".";
    PrintStatus();
    if (MemPool::IsInit()) {
      mempool_.PrintStatus();
      {
        std::ofstream handle{output_dir / "mempool.csv"};
        CHECK(handle.is_open());
        mempool_.DumpMemPool(handle);
        handle.close();
      }
    }

    {
      std::ofstream handle{output_dir / "entry_list.csv"};
      CHECK(handle.is_open());
      entry_list_.DumpMemEntryList(handle);
      handle.close();
    }
    {
      std::ofstream handle{output_dir / "freelist_small.csv"};
      CHECK(handle.is_open());
      free_list_small_.DumpFreeList(handle);
      handle.close();
    }
    {
      std::ofstream handle{output_dir / "freelist_large.csv"};
      CHECK(handle.is_open());
      free_list_large_.DumpFreeList(handle);
      handle.close();
    }
  }


  GenericAllocator(MemPool &mempool, Belong policy, size_t small_block_nbytes, bip::scoped_lock<bip::interprocess_mutex> &lock);

  virtual ~GenericAllocator();

  bool CheckState();

  void RegisterOOMHandler(std::function<void()> oom_handler) {
    oom_handler_ = oom_handler;
  }

  void PrintStatus() {
    enum class MemEntryType { kFreeLarge, kFreeSmall,  kTrainLarge, kTrainSmall, kNotAllocatedLarge, kNotAllocatedSmall, kUsed };
    std::unordered_map<MemEntryType, std::pair<std::vector<size_t>, std::string>> groupby = {
      {MemEntryType::kFreeLarge, std::make_pair(std::vector<size_t>{}, "FreeListLarge")},
      {MemEntryType::kFreeSmall, std::make_pair(std::vector<size_t>{}, "FreeListSmall")},
      {MemEntryType::kTrainLarge, std::make_pair(std::vector<size_t>{}, "TrainLarge")},
      {MemEntryType::kTrainSmall, std::make_pair(std::vector<size_t>{}, "TrainSmall")},
      {MemEntryType::kNotAllocatedLarge, std::make_pair(std::vector<size_t>{}, "NotAllocatedLarge")},
      {MemEntryType::kNotAllocatedSmall, std::make_pair(std::vector<size_t>{}, "NotAllocatedSmall")},
      {MemEntryType::kUsed, std::make_pair(std::vector<size_t>{}, "Used")},
    };
    for(auto *entry= entry_list_.GetEntry(0); entry !=nullptr; entry = entry_list_.GetNextEntry(entry)) {
      auto [index_begin, index_end] = GetAssociatedPhyMemIndex(entry);

      bool real_allocated = true;
      for(size_t k = index_begin; k < index_end; ++k) {
        if (mapped_mem_list_[k] == nullptr) { real_allocated = false; }
      }
      if (entry->is_free) {
        if (real_allocated && entry->is_small) {
          groupby.at(MemEntryType::kFreeSmall).first.push_back(entry->nbytes);
        } else if (real_allocated && !entry->is_small) {
          groupby.at(MemEntryType::kFreeLarge).first.push_back(entry->nbytes);
        } else if (!real_allocated && entry->is_small) {
          groupby.at(MemEntryType::kNotAllocatedSmall).first.push_back(entry->nbytes);
        } else if (!real_allocated && !entry->is_small) {
          groupby.at(MemEntryType::kNotAllocatedLarge).first.push_back(entry->nbytes);
        }
      } else {
        if (entry->is_train && entry->is_small) {
          groupby.at(MemEntryType::kTrainSmall).first.push_back(entry->nbytes);
        } else if (entry->is_train && !entry->is_small) {
          groupby.at(MemEntryType::kTrainLarge).first.push_back(entry->nbytes);
        } else if (!entry->is_train) {
          groupby.at(MemEntryType::kUsed).first.push_back(entry->nbytes);
        }

      }
    }
    for(auto &&[_, v] : groupby) {
      auto &&[arr, name] = v;
      LOG(INFO) << "------ " << name << "-----";
      if (arr.empty()) {
        LOG(INFO) << "empty array";
        continue;
      }
      LOG(INFO) << "max: " << ByteDisplay(*std::max_element(arr.cbegin(), arr.cend()));
      LOG(INFO) << "min: " << ByteDisplay(*std::min_element(arr.cbegin(), arr.cend()));
      LOG(INFO) << "avg: " << ByteDisplay(std::accumulate(arr.cbegin(), arr.cend(), 0UL) / arr.size());
      LOG(INFO) << "sum: " << ByteDisplay(std::accumulate(arr.cbegin(), arr.cend(), 0UL));
      LOG(INFO) << "cnt: " << arr.size();
    }
  }

  MemEntry *Split(MemEntry *entry, size_t addr_offset, size_t nbytes) {
    CHECK(entry != nullptr);
    CHECK(policy_ == Belong::kTrain || (policy_ == Belong::kInfer && !entry->is_train)) << entry;
    CHECK_GE(addr_offset, addr_offset) << entry;
    CHECK_LE(addr_offset + nbytes, entry->addr_offset + entry->nbytes) << entry;
    CHECK(!entry->is_free);
    LOG_IF(INFO, alloc_conf::VERBOSE) << log_prefix_ << "Split, addr_offset = " << addr_offset << ", nbytes = " << ByteDisplay(nbytes) << ", entry = " << entry <<  ".";
  
    if (entry->addr_offset < addr_offset) {
      auto *prev_entry = entry;
      entry = entry_list_.SplitEntry(prev_entry, addr_offset - entry->addr_offset);
      prev_entry->is_small = prev_entry->nbytes < small_block_nbytes_;
      prev_entry = (prev_entry->is_small ? free_list_small_ : free_list_large_).PushFreeEntry(prev_entry);

    }
    if (entry->addr_offset + entry->nbytes > addr_offset + nbytes) {
      auto *next_entry = entry_list_.SplitEntry(entry, addr_offset + nbytes - entry->addr_offset);
      next_entry->is_small = next_entry->nbytes < small_block_nbytes_;
      (next_entry->is_small ? free_list_small_ : free_list_large_).PushFreeEntry(next_entry);
    }
    entry->is_small = entry->nbytes < small_block_nbytes_;
    CHECK(!alloc_conf::ALWAYS_CHECK_STATE || CheckState());
    return entry;
  }

  MemEntry *MaybeMerge(MemEntry *entry) {
    if (!entry->is_alloc) { return entry; }
    CHECK(entry->is_small);
    bool put_free_list_large = true;
    size_t total_nbytes = entry->nbytes;
    MemEntry *prev_entry, *next_entry;
    if ((prev_entry = entry_list_.GetPrevEntry(entry)) != nullptr) {
      if (prev_entry->is_small) {
        CHECK(!prev_entry->is_free || !prev_entry->is_alloc) << prev_entry;
        put_free_list_large = false;
      } else {
        total_nbytes += prev_entry->nbytes;
      }
    }
    if ((next_entry = entry_list_.GetNextEntry(entry)) != nullptr) {
      if (next_entry->is_small) {
        CHECK(!next_entry->is_free || !next_entry->is_alloc) << next_entry;
        put_free_list_large = false;
      } else {
        total_nbytes += next_entry->nbytes;
      }
    }
    if (put_free_list_large && total_nbytes < small_block_nbytes_) {
      put_free_list_large = false;
    }
    if (put_free_list_large) {
      entry = free_list_small_.PopFreeEntry(entry);
      entry->is_small = false;
      entry = free_list_large_.PushFreeEntry(entry);
    }
    return entry;
  }

  // size_t GetCachedNBytes() {
  //   return cached_nbytes_;
  // }

  // size_t GetAllocatedNBytes() {
  //   return allocated_nbytes_;
  // }
};
}