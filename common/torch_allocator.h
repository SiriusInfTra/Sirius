#pragma once

#include "mempool.h"
#include <algorithm>
#include <cstddef>
#include <glog/logging.h>
#include <iterator>
#include <sys/types.h>
#include <map>
#include <ostream>
#include <list>

#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>

extern "C" {
void *col_malloc(ssize_t size, int device, cudaStream_t stream);

void col_free(void *ptr, ssize_t size, int device, cudaStream_t stream);
}

namespace colserve::sta {

namespace torch_allocator {

static const constexpr bool ALWAYS_CHECK_STATE = false;
static const constexpr bool STRICT_CHECK_STATE = false;
static const constexpr bool DUMP_BEFORE_CHECK = false;
static const constexpr bool VERBOSE = true;

}





struct TorchMemEntry;
using entry_linklist = std::list<TorchMemEntry*>;
using entry_addr_map = std::unordered_map<std::ptrdiff_t, TorchMemEntry*>;
using entry_nbytes_map = std::multimap<size_t, TorchMemEntry*, std::less<>>;

struct TorchMemEntry {
  std::ptrdiff_t  addr_offset;
  std::size_t     nbytes;
  bool            is_free;
  bool            is_small;

  entry_linklist::iterator    pos_entrylist;
  entry_addr_map::iterator    pos_entrytable;
  entry_nbytes_map::iterator  pos_freelist;
};

class EntryList {
private:
  entry_linklist entry_list_;
  entry_addr_map entry_by_addr;
public:
  void LinkNewEntry(TorchMemEntry *entry);

  TorchMemEntry *GetEntry(std::ptrdiff_t addr_offset);

  TorchMemEntry *GetPrevEntry(TorchMemEntry *entry);

  TorchMemEntry *GetNextEntry(TorchMemEntry *entry);

  TorchMemEntry *SplitFreeEntry(TorchMemEntry *origin_entry, size_t remain);

  TorchMemEntry *MergeFreeEntry(TorchMemEntry *first_entry,
                                TorchMemEntry *secound_entry);

  void DumpTorchMemEntryList(std::ostream &out);

  bool CheckState();
};

namespace pool {
  static const constexpr bool Small = true;
  static const constexpr bool Large = false;
}

inline std::pair<size_t, size_t> GetAssociatedPhyMemIndex(const TorchMemEntry *entry) {
  size_t index_begin = entry->addr_offset / MEM_BLOCK_NBYTES;
  size_t index_end = (entry->addr_offset + entry->nbytes - 1) / MEM_BLOCK_NBYTES + 1;
  return std::make_pair(index_begin, index_end);
}

template<bool is_small>
class FreeList {
private:
  EntryList& list_index_;
  std::multimap<size_t, TorchMemEntry*, std::less<>> entry_by_nbytes;
public:
  FreeList(EntryList &list_index);

  TorchMemEntry *PopFreeEntry(size_t nbytes);

  TorchMemEntry *PopFreeEntryLarge(size_t nbytes);

  void PushFreeEntry(TorchMemEntry *entry);

  void DumpFreeList(std::ostream &out);

  bool CheckState();
};

class TorchAllocator {
private:
  static std::unique_ptr<TorchAllocator> instance_;
  MemPool &mempool_;
  std::vector<PhyMem *> mapped_mem_list_;
  std::byte *base_ptr_;

  EntryList             entry_list_;
  FreeList<pool::Small> free_list_small_;
  FreeList<pool::Large> free_list_large_;

  void ExpandMemorySpace(const std::vector<PhyMem *> &phy_mem_list, size_t len);
  void ReleaseFreePhyMem() {
    LOG_IF(INFO, torch_allocator::VERBOSE) << "[TorchAllocator] Release free physical memory.";
    std::vector<int> ready_to_free_mask(mapped_mem_list_.size(), true);
    for (auto *entry = entry_list_.GetEntry(0); entry != nullptr; entry = entry_list_.GetNextEntry(entry)) {
      if (entry->is_free) { continue; }
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

      /* batch release physical memory page as it will acquire a lock */
      ready_to_free_mem.insert(ready_to_free_mem.cend(), mapped_mem_list_.cbegin() + k0, mapped_mem_list_.cbegin() + k);
      std::fill(mapped_mem_list_.begin() + k0, mapped_mem_list_.begin() + k, nullptr);
    }
    mempool_.DellocPhyMem(ready_to_free_mem);

    /* we can not pop last free physical memory entry unless we remove corresponding free entry */
    // while(!mapped_mem_list_.empty() && mapped_mem_list_.back() == nullptr) {
    //   mapped_mem_list_.pop_back();
    // }

    LOG_IF(INFO, torch_allocator::VERBOSE) << "[TorchAllocator] Free " << ready_to_free_mem.size() << " physical memory page(s).";
  }

  void EnsurePhyMemAlloc(TorchMemEntry *entry) {
    std::vector<size_t> missing_phy_mem_index_list;
    auto &&[index_begin, index_end] = GetAssociatedPhyMemIndex(entry);
    for (size_t k = index_begin; k < index_end; ++k) {
      if (mapped_mem_list_[k] == nullptr) {
        missing_phy_mem_index_list.push_back(k);
      }
    }
    LOG_IF(INFO, torch_allocator::VERBOSE && !missing_phy_mem_index_list.empty()) << "[TorchAllocator] missing " << missing_phy_mem_index_list.size() << " physical memory page(s), try allocate.";
    std::vector<PhyMem *> request_phy_mem_list(missing_phy_mem_index_list.size());
    CHECK_EQ(mempool_.AllocPhyMem(request_phy_mem_list, Belong::kTrain), request_phy_mem_list.size());
    for (size_t k = 0; k < request_phy_mem_list.size(); ++k) {
      size_t phy_mem_index = missing_phy_mem_index_list[k];
      mapped_mem_list_[phy_mem_index] = request_phy_mem_list[k];
      std::byte *target_addr = base_ptr_ + phy_mem_index * MEM_BLOCK_NBYTES;
      CU_CALL(cuMemMap(reinterpret_cast<CUdeviceptr>(target_addr), MEM_BLOCK_NBYTES, 0, mapped_mem_list_[phy_mem_index]->cu_handle, 0));
    }
  }
  
public:
  static TorchAllocator &Get();

  TorchAllocator(MemPool &mempool);

  ~TorchAllocator();

  bool CheckState();

  template<bool retry_alloc>
  std::byte *Alloc(size_t unaligned_nbytes);

  void Free(std::byte *ptr);

  void EmptyCache() {
    ReleaseFreePhyMem();
  }
};
}