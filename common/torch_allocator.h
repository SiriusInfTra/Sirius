#pragma once

#include "generic_allocator.h"
#include "mempool.h"
#include <algorithm>
#include <cstddef>
#include <glog/logging.h>
#include <iterator>
#include <sys/types.h>
#include <map>
#include <mutex>
#include <ostream>
#include <list>

#include <set>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace colserve::sta {

class TorchAllocator : public GenericAllocator {
private:
  static std::unique_ptr<TorchAllocator> instance_;
  std::mutex mutex_;
public:
  static TorchAllocator &Get();
  TorchAllocator(MemPool &mempool);

  std::set<void *> allocated_ptrs;

  std::byte *Alloc(size_t unaligned_nbytes, bool retry_alloc) {
    std::unique_lock lock(mutex_);
    if (unaligned_nbytes == 0) { return nullptr; }
    size_t aligned_nbytes = detail::AlignedNBytes<512_B>(unaligned_nbytes);
    bool is_small = aligned_nbytes < 1_MB;
    LOG_IF(INFO, torch_allocator::VERBOSE) << "[TorchAllocator] Alloc " << detail::ByteDisplay(unaligned_nbytes) << ", (aligned to " << detail::ByteDisplay(aligned_nbytes) << ").";
    MemEntry *free_entry = nullptr;
    if (is_small) {
      free_entry = free_list_small_.PopFreeEntry(aligned_nbytes);
      if (free_entry == nullptr) {
        /* try to pop entry from large */
        auto *free_entry_large =
            free_list_large_.PopFreeEntry(MEM_BLOCK_NBYTES);
        if (free_entry_large != nullptr) {
          free_entry_large->is_small = true;
          free_list_small_.PushFreeEntry(free_entry_large);
          free_entry = free_list_small_.PopFreeEntry(aligned_nbytes);
        }
      }
    } else {
      free_entry = free_list_large_.PopFreeEntry(aligned_nbytes);
    }
    if (retry_alloc) {
      if (free_entry == nullptr) {
        /* try to allocate from global mempool*/
        std::vector<PhyMem *> phy_mem_list{
            detail::AlignedNBytes<MEM_BLOCK_NBYTES * 8>(aligned_nbytes) /
            MEM_BLOCK_NBYTES};
        size_t allocated = mempool_.AllocPhyMem(phy_mem_list, policy_);
        LOG_IF(INFO, allocated < phy_mem_list.size())
            << "[Torch Allocator] Require " << phy_mem_list.size()
            << " physical memory blocks, but only get " << allocated << ", later retry may fail.";
        CHECK_GT(allocated, 0);
        auto *expand_entry = new MemEntry;
        expand_entry->addr_offset = MEM_BLOCK_NBYTES * mapped_mem_list_.size();
        expand_entry->nbytes = allocated * MEM_BLOCK_NBYTES;
        expand_entry->is_free = false;
        expand_entry->is_small = false;
        ExpandMemorySpace(phy_mem_list, allocated);
        entry_list_.LinkNewEntry(expand_entry);
        free_list_large_.PushFreeEntry(expand_entry);
        CHECK(!torch_allocator::ALWAYS_CHECK_STATE || CheckState());
        lock.unlock();
        return Alloc(unaligned_nbytes, false);
      }
    }
    if (free_entry == nullptr) {
      LOG(INFO) << "[TorchAllocator] OMM";
      PrintOnCrash();
    }
    
    CHECK(free_entry != nullptr);
    CHECK(!free_entry->is_free);
    CHECK(!torch_allocator::ALWAYS_CHECK_STATE || CheckState());
    EnsurePhyMemAlloc(free_entry);
    allocated_nbytes_ += free_entry->nbytes;
    CHECK(allocated_ptrs.insert(base_ptr_ + free_entry->addr_offset).second == true) << "err1";
    LOG_IF(INFO, torch_allocator::VERBOSE) << "[TorchAllocator] Alloc " << detail::ByteDisplay(unaligned_nbytes) << ", (aligned to " << detail::ByteDisplay(aligned_nbytes) << "): " << base_ptr_ + free_entry->addr_offset << ". ";
    
    return base_ptr_ + free_entry->addr_offset;
  }

  void Free(std::byte *ptr) {
    std::unique_lock lock(mutex_);
    if (ptr == nullptr) { return; }
    CHECK_GE(ptr, base_ptr_);
    CHECK_LT(ptr, base_ptr_ + mapped_mem_list_.size() * MEM_BLOCK_NBYTES);
    auto addr_offset = ptr - base_ptr_;
    auto entry = entry_list_.GetEntry(addr_offset);
    size_t aligned_nbytes = entry->nbytes;
    LOG_IF(INFO, torch_allocator::VERBOSE) << "[TorchAllocator] Free, nbytes = " << detail::ByteDisplay(entry->nbytes) << "ptr = " << ptr << ".";
    CHECK(entry != nullptr);
    CHECK(!entry->is_free);
    (entry->is_small ? free_list_small_ : free_list_large_).PushFreeEntry(entry);
    allocated_nbytes_ -= aligned_nbytes;
    allocated_ptrs.erase(ptr);
    CHECK(!torch_allocator::ALWAYS_CHECK_STATE || CheckState());
    
  }

};



}