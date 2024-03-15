#pragma once
#include "generic_allocator.h"
#include "mempool.h"
#include "util.h"
#include <algorithm>
#include <cstddef>
#include <glog/logging.h>
#include <iterator>
#include <memory>
#include <mutex>
#include <sys/types.h>
#include <map>
#include <ostream>
#include <list>

#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h> 

namespace colserve::sta {

class TVMAllocator : public GenericAllocator {
private:
  static std::unique_ptr<TVMAllocator> instance_;
  std::mutex mutex_;
public:
  static TVMAllocator &Get();
  TVMAllocator(MemPool &mempool);

  MemEntry *AllocPhyMem(size_t n, bool require) {
    std::vector<PhyMem*> allocated_pages(n);
    size_t allocated = mempool_.AllocPhyMem(allocated_pages, policy_);
    if ((require && allocated < n) || (!require && allocated == 0)) {
      LOG(INFO) << "[TVMAllocator] Fail to alloc phymem, only get " << allocated << ".";
      DumpState();
      LOG(FATAL) << "PhyMem OOM";
    }
    if (auto *free_va = free_list_large_.PopFreeEntry(allocated * MEM_BLOCK_NBYTES); free_va) {
      for (size_t k = 0; k < free_va->nbytes / MEM_BLOCK_NBYTES; ++k) {
        mapped_mem_list_[free_va->addr_offset / MEM_BLOCK_NBYTES + k] = allocated_pages[k];
        CU_CALL(cuMemMap(reinterpret_cast<CUdeviceptr>(base_ptr_ + free_va->addr_offset + k * MEM_BLOCK_NBYTES), MEM_BLOCK_NBYTES, 0, allocated_pages[k]->cu_handle, 0));
        // cached_nbytes_ += MEM_BLOCK_NBYTES;
      }
      CUmemAccessDesc acc_desc = {
        .location = {.type = CU_MEM_LOCATION_TYPE_DEVICE, .id = 0},
        .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE};
      CU_CALL(cuMemSetAccess(reinterpret_cast<CUdeviceptr>(base_ptr_ + free_va->addr_offset), free_va->nbytes, &acc_desc, 1));
      return free_va;
    } else {
      auto *expand_entry = new MemEntry;
      expand_entry->addr_offset = MEM_BLOCK_NBYTES * mapped_mem_list_.size();
      expand_entry->nbytes = allocated * MEM_BLOCK_NBYTES;
      expand_entry->is_free = false;
      expand_entry->is_small = true;
      ExpandMemorySpace(allocated_pages, allocated);
      entry_list_.LinkNewEntry(expand_entry);
      return expand_entry;
    }
  }

  std::byte *Alloc(size_t unaligned_nbytes, bool retry_alloc) {
    std::unique_lock lock{mutex_};
    if (unaligned_nbytes == 0) { return nullptr; }
    bool is_small = unaligned_nbytes < MEM_BLOCK_NBYTES;
    size_t aligned_nbytes = detail::AlignedNBytes<512_B>(unaligned_nbytes);
    
    LOG_IF(INFO, torch_allocator::VERBOSE) << "[TorchAllocator] Alloc " << detail::ByteDisplay(unaligned_nbytes) << ", (aligned to " << detail::ByteDisplay(aligned_nbytes) << ").";
    MemEntry *free_entry = nullptr;
    if (is_small) {
      free_entry = free_list_small_.PopFreeEntry(aligned_nbytes);
      if (free_entry == nullptr) {
        auto *allocated_entry = AllocPhyMem(1, true);
        allocated_entry->is_small = true;
        CHECK_LE(allocated_entry->nbytes, 2 * MEM_BLOCK_NBYTES);
        free_list_small_.PushFreeEntry(allocated_entry);
        free_entry = free_list_small_.PopFreeEntry(aligned_nbytes);
        CHECK(free_entry != nullptr);
        {
          auto *entry = free_list_small_.PopFreeEntryLarge(MEM_BLOCK_NBYTES);
          if (entry != nullptr) {
            auto &&[entry1, succ] = MaybeRelease(entry);
            if (succ) {
              entry1->is_small = false;
              free_list_large_.PushFreeEntry(entry1);
            } else {
              entry1->is_small = true;
              free_list_small_.PushFreeEntry(entry1);
            }
          }
        }
      } 
    } else {
      auto *allocated_entry = AllocPhyMem(detail::AlignedNBytes<MEM_BLOCK_NBYTES>(unaligned_nbytes) / MEM_BLOCK_NBYTES, true);
      allocated_entry->is_small = true;
      free_list_small_.PushFreeEntry(allocated_entry);
      free_entry = free_list_small_.PopFreeEntry(aligned_nbytes);
      {
        auto *entry = free_list_small_.PopFreeEntryLarge(MEM_BLOCK_NBYTES);
        if (entry != nullptr) {
          auto &&[entry1, succ] = MaybeRelease(entry);
          if (succ) {
            entry1->is_small = false;
            free_list_large_.PushFreeEntry(entry1);
          } else {
            entry1->is_small = true;
            free_list_small_.PushFreeEntry(entry1);
          }
        }
      }
    }
    CHECK(free_entry != nullptr);
    CHECK(!torch_allocator::ALWAYS_CHECK_STATE || CheckState());
    mempool_.AddAllocatedNbytes(free_entry->nbytes, policy_);
    return base_ptr_ + free_entry->addr_offset;
  }

  std::pair<MemEntry*, bool> MaybeRelease(MemEntry *entry) {
    size_t aligned_begin = (entry->addr_offset + MEM_BLOCK_NBYTES - 1) / MEM_BLOCK_NBYTES * MEM_BLOCK_NBYTES;
    size_t aligned_end = (entry->addr_offset + entry->nbytes) / MEM_BLOCK_NBYTES * MEM_BLOCK_NBYTES;
    if (aligned_begin < aligned_end) {
      if (aligned_begin > entry->addr_offset) {
        auto *next_entry = entry_list_.SplitEntry(entry, aligned_begin - entry->addr_offset);
        CHECK_LE(entry->nbytes, MEM_BLOCK_NBYTES);
        free_list_small_.PushFreeEntry(entry);
        entry = next_entry;
      }
      if (aligned_end < entry->addr_offset + entry->nbytes) {
        auto *next_entry = entry_list_.SplitEntry(entry, aligned_end - aligned_begin);
        CHECK_LE(next_entry->nbytes, MEM_BLOCK_NBYTES);
        free_list_small_.PushFreeEntry(next_entry);
      }
      CHECK_EQ(entry->addr_offset, aligned_begin);
      CHECK_EQ(entry->addr_offset + entry->nbytes, aligned_end);
      std::vector<PhyMem *> free_mem;
      for (size_t k = aligned_begin / MEM_BLOCK_NBYTES; k < aligned_end / MEM_BLOCK_NBYTES; ++k) {
        CHECK(mapped_mem_list_[k] != nullptr);
        free_mem.push_back(mapped_mem_list_[k]);
        mapped_mem_list_[k] = nullptr;
      }
      CU_CALL(cuMemUnmap(reinterpret_cast<CUdeviceptr>(base_ptr_ + entry->addr_offset), entry->nbytes));
      // cached_nbytes_ -= entry->nbytes;
      mempool_.DeallocPhyMem(free_mem);

      return {entry, true};
    }
      return {entry, false};
  }

  void Free(std::byte *ptr) {
    std::unique_lock lock{mutex_};
    if (ptr == nullptr) { return; }
    CHECK_GE(ptr, base_ptr_);
    CHECK_LT(ptr, base_ptr_ + mapped_mem_list_.size() * MEM_BLOCK_NBYTES);
    auto addr_offset = ptr - base_ptr_;
    auto entry = entry_list_.GetEntry(addr_offset);
    CHECK(entry != nullptr);
    CHECK(!entry->is_free);
    CHECK(entry->is_small);
    size_t aligned_nbytes = entry->nbytes;
    LOG_IF(INFO, torch_allocator::VERBOSE) << "[TVMAllocator] Free, nbytes = " << detail::ByteDisplay(entry->nbytes) << ".";
    entry = free_list_small_.PushFreeEntry(entry);
    free_list_small_.PopFreeEntry(entry);
    if (auto &&[entry1, succ] = MaybeRelease(entry); succ) {
      entry1->is_small = false;
      free_list_large_.PushFreeEntry(entry1);
    } else {
      entry1->is_small = true;
      free_list_small_.PushFreeEntry(entry1);
    }
    mempool_.SubAllocatedNbytes(aligned_nbytes, policy_);
    CHECK(!torch_allocator::ALWAYS_CHECK_STATE || CheckState());
  }
};

  
}