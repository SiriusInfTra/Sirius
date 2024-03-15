#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <list>
#include <map>
#include <numeric>
#include <ostream>
#include <unordered_map>
#include <functional>
#include <fstream>
#include "mempool.h"
#include "util.h"
#include <glog/logging.h>
#include <utility>

namespace colserve::sta {

static const constexpr size_t VA_RESERVE_SCALE = 16;

namespace torch_allocator {

static const constexpr bool ALWAYS_CHECK_STATE = true;
static const constexpr bool STRICT_CHECK_STATE = true;
static const constexpr bool DUMP_BEFORE_CHECK = false;
// static const constexpr bool VERBOSE = true;
static const constexpr bool VERBOSE = false;

}



struct MemEntry;
using entry_linklist = std::list<MemEntry*>;
using entry_addr_map = std::unordered_map<std::ptrdiff_t, MemEntry*>;
using entry_nbytes_map = std::multimap<size_t, MemEntry*, std::less<>>;

struct MemEntry {
  std::ptrdiff_t  addr_offset;
  std::size_t     nbytes;
  bool            is_free;
  bool            is_small;

  entry_linklist::iterator    pos_entrylist;
  entry_addr_map::iterator    pos_entrytable;
  entry_nbytes_map::iterator  pos_freelist;
};

inline std::pair<size_t, size_t> GetAssociatedPhyMemIndex(const MemEntry *entry) {
  size_t index_begin = entry->addr_offset / MEM_BLOCK_NBYTES;
  size_t index_end = (entry->addr_offset + entry->nbytes + MEM_BLOCK_NBYTES - 1) / MEM_BLOCK_NBYTES;
  return std::make_pair(index_begin, index_end);
}


class EntryList {
private:
  entry_linklist entry_list_;
  entry_addr_map entry_by_addr;
public:
  void LinkNewEntry(MemEntry *entry);

  MemEntry *GetEntry(std::ptrdiff_t addr_offset);

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
  EntryList& list_index_;
  const bool is_small_;
  const Belong policy_;
  std::multimap<size_t, MemEntry*, std::less<>> entry_by_nbytes;
public:


  FreeList(EntryList &list_index, bool is_small, Belong policy);

  MemEntry *PopFreeEntry(size_t nbytes);

  MemEntry *PopFreeEntry(MemEntry *free_entry);

  MemEntry *PopFreeEntryLarge(size_t nbytes);

  MemEntry* PushFreeEntry(MemEntry *entry);

  void DumpFreeList(std::ostream &out);

  bool CheckState();
};



class GenericAllocator {
protected:
  MemPool &mempool_;
  std::vector<PhyMem *> mapped_mem_list_;
  // std::atomic<size_t> cached_nbytes_;
  std::byte *base_ptr_;

  EntryList             entry_list_;
  FreeList free_list_small_;
  FreeList free_list_large_;
  // std::atomic<size_t> allocated_nbytes_;
  std::function<void()>       oom_handler_;


  const Belong policy_;

  void ExpandMemorySpace(const std::vector<PhyMem *> &phy_mem_list, size_t len);
  void ReleaseFreePhyMem() {
    LOG_IF(INFO, torch_allocator::VERBOSE) << "[TorchAllocator] Release free physical memory.";
    size_t allocated_nbytes = 0;
    std::vector<int> ready_to_free_mask(mapped_mem_list_.size(), true);
    for (auto *entry = entry_list_.GetEntry(0); entry != nullptr; entry = entry_list_.GetNextEntry(entry)) {
      if (entry->is_free) { continue; }
      allocated_nbytes += entry->nbytes;
      auto &&[index_begin, index_end] = GetAssociatedPhyMemIndex(entry);
      std::fill(ready_to_free_mask.begin() + index_begin, ready_to_free_mask.begin() + index_end, false);
    }
    // some physical memory pages already free
    for (size_t k = 0; k < ready_to_free_mask.size(); ++k) {
      if (mapped_mem_list_[k] == nullptr) { ready_to_free_mask[k] = false; }
    }
    std::vector<PhyMem *> ready_to_free_mem;
    for (size_t k = 0; k < ready_to_free_mask.size(); ++k) {
      if (ready_to_free_mask[k] == false) { continue; }
      size_t k0 = k;
      while (++k < ready_to_free_mask.size() && ready_to_free_mask[k] == true) {}

      CU_CALL(cuMemUnmap(reinterpret_cast<CUdeviceptr>(base_ptr_ + k0 * MEM_BLOCK_NBYTES), (k - k0) * MEM_BLOCK_NBYTES));
      // cached_nbytes_ -= (k - k0) * MEM_BLOCK_NBYTES;
      /* batch release physical memory page as it will acquire a lock */
      ready_to_free_mem.insert(ready_to_free_mem.cend(), mapped_mem_list_.cbegin() + k0, mapped_mem_list_.cbegin() + k);
      std::fill(mapped_mem_list_.begin() + k0, mapped_mem_list_.begin() + k, nullptr);
    }
    mempool_.DeallocPhyMem(ready_to_free_mem);
    size_t physical_nbytes = std::count_if(mapped_mem_list_.cbegin(), mapped_mem_list_.cend(), [](auto *ptr) { return ptr != nullptr; }) * MEM_BLOCK_NBYTES;
    /* we can not pop last free physical memory entry unless we remove corresponding free entry */
    // while(!mapped_mem_list_.empty() && mapped_mem_list_.back() == nullptr) {
    //   mapped_mem_list_.pop_back();
    // }

    LOG(INFO) << "[TorchAllocator] Free " << ready_to_free_mem.size() << " physical memory page(s),"
      << " current allocated: " << detail::ByteToMB(allocated_nbytes) 
      << " current physical: " << detail::ByteToMB(physical_nbytes) << ".";
  }

  void EnsurePhyMemAlloc(MemEntry *entry) {
    std::vector<size_t> missing_phy_mem_index_list;
    auto &&[index_begin, index_end] = GetAssociatedPhyMemIndex(entry);
    for (size_t k = index_begin; k < index_end; ++k) {
      if (mapped_mem_list_[k] == nullptr) {
        missing_phy_mem_index_list.push_back(k);
      }
    }
    LOG_IF(INFO, torch_allocator::VERBOSE && !missing_phy_mem_index_list.empty()) << "[TorchAllocator] missing " << missing_phy_mem_index_list.size() << " physical memory page(s), try allocate.";
    std::vector<PhyMem *> request_phy_mem_list(missing_phy_mem_index_list.size());
    if (mempool_.AllocPhyMem(request_phy_mem_list, policy_) != request_phy_mem_list.size()) {
      DumpState();
      LOG(FATAL) << "OOM";
    }
    // CHECK_EQ(mempool_.AllocPhyMem(request_phy_mem_list, Belong::kTrain), request_phy_mem_list.size());
    for (size_t k = 0; k < request_phy_mem_list.size(); ++k) {
      size_t phy_mem_index = missing_phy_mem_index_list[k];
      mapped_mem_list_[phy_mem_index] = request_phy_mem_list[k];
      std::byte *target_addr = base_ptr_ + phy_mem_index * MEM_BLOCK_NBYTES;
      CU_CALL(cuMemMap(reinterpret_cast<CUdeviceptr>(target_addr), MEM_BLOCK_NBYTES, 0, mapped_mem_list_[phy_mem_index]->cu_handle, 0));
      CUmemAccessDesc acc_desc = {
        .location = {.type = CU_MEM_LOCATION_TYPE_DEVICE, .id = 0},
        .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE};
      CU_CALL(cuMemSetAccess(reinterpret_cast<CUdeviceptr>(target_addr), MEM_BLOCK_NBYTES, &acc_desc, 1));
      // cached_nbytes_ += MEM_BLOCK_NBYTES;
    }
  }
  
  void DumpState() {
    if (oom_handler_ != nullptr) {
      oom_handler_();
    }
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << "./" << "mempool_dump_" << policy_ << "_" << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%H-%M-%S") << "_" << ms.count();
    std::filesystem::path output_dir{ss.str()};
    CHECK(std::filesystem::create_directory(output_dir));
    LOG(INFO) << "[GenericAllocator] Dump has been written to: " << ss.str() << ".";
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
public:
  static GenericAllocator &Get();

  GenericAllocator(MemPool &mempool, Belong policy);

  virtual ~GenericAllocator();

  bool CheckState();

  void RegisterOOMHandler(std::function<void()> oom_handler) {
    oom_handler_ = oom_handler;
  }

  void PrintStatus() {
    enum class MemEntryType { kFreeLarge, kFreeSmall, kUsed };
    std::unordered_map<MemEntryType, std::pair<std::vector<size_t>, std::string>> groupby = {
      {MemEntryType::kFreeLarge, std::make_pair(std::vector<size_t>{}, "FreeListLarge")},
      {MemEntryType::kFreeSmall, std::make_pair(std::vector<size_t>{}, "FreeListSmall")},
      {MemEntryType::kUsed, std::make_pair(std::vector<size_t>{}, "Used")},
    };
    for(auto *entry= entry_list_.GetEntry(0); entry !=nullptr; entry = entry_list_.GetNextEntry(entry)) {
      auto [index_begin, index_end] = GetAssociatedPhyMemIndex(entry);
      bool real_allocated = true;
      for(size_t k = index_begin; k < index_end; ++k) {
        if (mapped_mem_list_[k] == nullptr) { real_allocated = false; }
      }
      if (!real_allocated) { continue; }
      if (entry->is_free) {
        if (entry->is_small) {
          groupby.at(MemEntryType::kFreeSmall).first.push_back(entry->nbytes);
        } else {
          groupby.at(MemEntryType::kFreeLarge).first.push_back(entry->nbytes);
        }
      } else {
        groupby.at(MemEntryType::kUsed).first.push_back(entry->nbytes);
      }
    }
    for(auto &&[_, v] : groupby) {
      auto &&[arr, name] = v;
      LOG(INFO) << "------ " << name << "-----";
      if (arr.empty()) {
        LOG(INFO) << "empty array";
        continue;
      }
      LOG(INFO) << "max: " << detail::ByteDisplay(*std::max_element(arr.cbegin(), arr.cend()));
      LOG(INFO) << "min: " << detail::ByteDisplay(*std::min_element(arr.cbegin(), arr.cend()));
      LOG(INFO) << "avg: " << detail::ByteDisplay(std::accumulate(arr.cbegin(), arr.cend(), 0UL) / arr.size());
      LOG(INFO) << "sum: " << detail::ByteDisplay(std::accumulate(arr.cbegin(), arr.cend(), 0UL));
      LOG(INFO) << "cnt: " << arr.size();
    }
  }

  void EmptyCache() {
    ReleaseFreePhyMem();
  }

  // size_t GetCachedNBytes() {
  //   return cached_nbytes_;
  // }

  // size_t GetAllocatedNBytes() {
  //   return allocated_nbytes_;
  // }
};
}