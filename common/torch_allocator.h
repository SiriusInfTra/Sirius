#pragma once

#include <common/generic_allocator.h>
#include <common/mempool.h>
#include <common/tvm_allocator.h>
#include <common/util.h>

#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

#include <atomic>
#include <glog/logging.h>
#include <sys/types.h>
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <map>
#include <mutex>
#include <ostream>
#include <list>
#include <set>
#include <utility>
#include <vector>


namespace colserve::sta {

class TorchAllocator : public GenericAllocator {
public:
  static const constexpr size_t ALIGN_NBYTES = 512_B; 

private:
  static std::unique_ptr<TorchAllocator> instance_;
  std::vector<std::pair<CUdeviceptr, size_t>> planning_unmap_;
  std::atomic<size_t> *peek_allocated_nbytes_;

  TorchAllocator(MemPool &mempool, bip::scoped_lock<bip::interprocess_mutex> &lock);

  void EnsureUnmap() {
    if constexpr (alloc_conf::DELAY_UNMAP) {
      for (auto &&[dev_ptr, sz] : planning_unmap_) {
        CU_CALL(cuMemUnmap(reinterpret_cast<CUdeviceptr>(dev_ptr), sz));
      }
      planning_unmap_.clear();
    }
    CHECK(planning_unmap_.empty());
  }

  void ReleaseFreePhyMem(bip::scoped_lock<bip::interprocess_mutex> &lock) {
    LOG_IF(INFO, alloc_conf::VERBOSE) << log_prefix_ << "Release free physical memory.";
    size_t allocated_nbytes = 0;
    std::vector<int> ready_to_free_mask(mapped_mem_list_.size(), true);
    entry_list_.IterateRange(0, mempool_.mempool_nbytes, [&](MemEntry *entry){
      if (!entry->is_free) {
        allocated_nbytes += entry->nbytes;
        auto &&[index_begin, index_end] = GetAssociatedPhyMemIndex(entry);
        std::fill(ready_to_free_mask.begin() + index_begin, ready_to_free_mask.begin() + index_end, false);
      }
      return entry;
    });
    LOG(INFO) << "ReleaseFreePhyMem 1";
    // some physical memory pages already free
    for (size_t k = 0; k < ready_to_free_mask.size(); ++k) {
      if (mapped_mem_list_[k] == nullptr) { 
        ready_to_free_mask[k] = false; 
      }
    }


    std::vector<PhyMem *> ready_to_free_mem;
    for (size_t k = 0; k < ready_to_free_mask.size(); ++k) {
      if (ready_to_free_mask[k] == false) { continue; }
      // find continuous phy pages to free
      size_t k0 = k;
      while (++k < ready_to_free_mask.size() && ready_to_free_mask[k] == true) {}
      if constexpr (alloc_conf::DELAY_UNMAP) {
        planning_unmap_.emplace_back(
            reinterpret_cast<CUdeviceptr>(base_ptr_ + k0 * MEM_BLOCK_NBYTES), 
            (k - k0) * MEM_BLOCK_NBYTES);
      } else {
        CU_CALL(cuMemUnmap(
            reinterpret_cast<CUdeviceptr>(base_ptr_ + k0 * MEM_BLOCK_NBYTES), 
            (k - k0) * MEM_BLOCK_NBYTES));
      }
      // cached_nbytes_ -= (k - k0) * MEM_BLOCK_NBYTES;
      /* batch release physical memory page as it will acquire a lock */
      ready_to_free_mem.insert(ready_to_free_mem.cend(), 
                               mapped_mem_list_.cbegin() + k0, 
                               mapped_mem_list_.cbegin() + k);
      std::fill(mapped_mem_list_.begin() + k0, mapped_mem_list_.begin() + k, nullptr);
    }
    entry_list_.IterateRange(0, mempool_.mempool_nbytes, [&](MemEntry *entry){
      bool is_alloc = entry->is_alloc;
      entry_list_.UpdateAllocFlag(entry);
      CHECK((is_alloc == false && entry->is_alloc == false) || is_alloc == true) << entry << " " << is_alloc;
      return entry;
    });
    LOG(INFO) << "ReleaseFreePhyMem 2";
    mempool_.DeallocPhyMem(ready_to_free_mem);
    TVMAllocator::Get().SyncFreeTrain(ready_to_free_mem, lock);
    size_t physical_nbytes = std::count_if(
        mapped_mem_list_.cbegin(), 
        mapped_mem_list_.cend(), 
        [](auto *ptr) { return ptr != nullptr; }
      ) * MEM_BLOCK_NBYTES;

    /* we can not pop last free physical memory entry unless we remove corresponding free entry */
    // while(!mapped_mem_list_.empty() && mapped_mem_list_.back() == nullptr) {
    //   mapped_mem_list_.pop_back();
    // }

    LOG_IF(INFO, alloc_conf::VERBOSE) << log_prefix_ 
        << "Free " << ready_to_free_mem.size() << " physical memory page(s),"
        << " current allocated: " << ByteToMB(allocated_nbytes) 
        << " current physical: " << ByteToMB(physical_nbytes) << ".";
  }

  void EnsurePhyMemAlloc(MemEntry *entry, bip::scoped_lock<bip::interprocess_mutex> &lock) {
    // entry_list_.IterateRange(0, mempool_.mempool_nbytes, [&](MemEntry *entry){
    //   bool is_alloc = entry->is_alloc;
    //   entry_list_.UpdateAllocFlag(entry);
    //   CHECK_EQ(is_alloc, entry->is_alloc) << entry << "error1";
    //   return entry;
    // });
    std::vector<size_t> missing_phy_mem_index_list;
    auto &&[index_begin, index_end] = GetAssociatedPhyMemIndex(entry);
    for (size_t k = index_begin; k < index_end; ++k) {
      if (mapped_mem_list_[k] == nullptr) {
        missing_phy_mem_index_list.push_back(k);
      }
    }
    if (missing_phy_mem_index_list.empty()) { 
      return; 
    }

    LOG_IF(INFO, alloc_conf::VERBOSE) << log_prefix_ 
        << "missing " << missing_phy_mem_index_list.size() 
        << " physical memory page(s), try allocate.";
    EnsureUnmap();
    std::vector<PhyMem *> request_phy_mem_list = mempool_.AllocNumPhyMem( policy_, missing_phy_mem_index_list.size());
    if (request_phy_mem_list.size() < missing_phy_mem_index_list.size()) {
      DumpState();
      TVMAllocator::Get().DumpState();
      LOG(FATAL) << log_prefix_ << "OOM while alloc phy mem page, entry = " << entry << ".";
    }
    TVMAllocator::Get().SyncAllocTrain(request_phy_mem_list, lock);
    for (size_t k = 0; k < request_phy_mem_list.size(); ++k) {
      size_t phy_mem_index = missing_phy_mem_index_list[k];
      mapped_mem_list_[phy_mem_index] = request_phy_mem_list[k];
      std::byte *target_addr = base_ptr_ + phy_mem_index * MEM_BLOCK_NBYTES;
      CU_CALL(cuMemMap(
          reinterpret_cast<CUdeviceptr>(target_addr), 
          MEM_BLOCK_NBYTES, 0, mapped_mem_list_[phy_mem_index]->cu_handle, 0));
      CUmemAccessDesc acc_desc = {
        .location = {.type = CU_MEM_LOCATION_TYPE_DEVICE, .id = 0},
        .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE};
      CU_CALL(cuMemSetAccess(reinterpret_cast<CUdeviceptr>(target_addr), MEM_BLOCK_NBYTES, &acc_desc, 1));
    }
    entry->is_alloc = true;
    for (auto *prev_entry = entry_list_.GetPrevEntry(entry); prev_entry && prev_entry->is_free && !prev_entry->is_alloc; prev_entry = entry_list_.GetPrevEntry(prev_entry)) {
      entry_list_.UpdateAllocFlag(prev_entry);
      if (prev_entry->is_alloc) {
        if (prev_entry->is_small) {
          free_list_small_.PopFreeEntry(prev_entry);
          prev_entry = free_list_small_.PushFreeEntry(prev_entry);
          prev_entry = MaybeMerge(prev_entry);
        } else {
          free_list_large_.PopFreeEntry(prev_entry);
          prev_entry = free_list_large_.PushFreeEntry(prev_entry);
        }
      } else {
        break;
      }
    }
    for (auto *next_entry = entry_list_.GetNextEntry(entry); next_entry && next_entry->is_free && !next_entry->is_alloc; next_entry = entry_list_.GetNextEntry(next_entry)) {
      entry_list_.UpdateAllocFlag(next_entry);
      if (next_entry->is_alloc) {
        if (next_entry->is_small) {
          free_list_small_.PopFreeEntry(next_entry);
          next_entry = free_list_small_.PushFreeEntry(next_entry);
          next_entry = MaybeMerge(next_entry);
        } else {
          free_list_large_.PopFreeEntry(next_entry);
          next_entry = free_list_large_.PushFreeEntry(next_entry);
        }
      } else {
        break;
      }
    }
    // entry_list_.IterateRange(0, mempool_.mempool_nbytes, [&](MemEntry *entry){
    //   bool is_alloc = entry->is_alloc;
    //   entry_list_.UpdateAllocFlag(entry);
    //   CHECK_EQ(is_alloc, entry->is_alloc) << entry;
    //   return entry;
    // });
  }

  MemEntry *Free0(MemEntry *entry, bip::scoped_lock<bip::interprocess_mutex> &lock) {
    CHECK(entry != nullptr);
    CHECK(!entry->is_free);
    size_t nbytes = entry->nbytes;
    LOG_IF(INFO, alloc_conf::VERBOSE) << log_prefix_ 
        << "Free, nbytes = " << ByteDisplay(entry->nbytes) 
        << "ptr = " << base_ptr_ + entry->addr_offset << ".";
    
    if (entry->is_small) {
      entry = free_list_small_.PushFreeEntry(entry);
      entry = MaybeMerge(entry);
    } else {
      entry = free_list_large_.PushFreeEntry(entry);
      auto *prev_entry_nbytes = entry_list_.GetPrevEntry(entry);
      if (prev_entry_nbytes && prev_entry_nbytes->is_small && prev_entry_nbytes->is_free) {
        size_t prev_nbytes = prev_entry_nbytes->nbytes;
        auto *maybe_merged_entry = MaybeMerge(prev_entry_nbytes);
        if (maybe_merged_entry->nbytes > prev_nbytes) {
          entry = maybe_merged_entry;
        }
      }
      auto *next_entry = entry_list_.GetNextEntry(entry);
      if (next_entry && next_entry->is_small && next_entry->is_free) {
        size_t next_entry_nbytes = next_entry->nbytes;
        auto *maybe_merged_entry = MaybeMerge(next_entry);
        if (maybe_merged_entry->nbytes > next_entry_nbytes) {
          entry = maybe_merged_entry;
        }
      }
    }
  
    mempool_.SubAllocatedNbytes(nbytes, policy_);
    CHECK(!alloc_conf::ALWAYS_CHECK_STATE || CheckState());
    return entry;
  }

  MemEntry *Alloc0(size_t nbytes, bool retry_alloc, bip::scoped_lock<bip::interprocess_mutex> &lock) {
    CHECK_GT(nbytes, 0);
    LOG_IF(INFO, alloc_conf::VERBOSE) << log_prefix_ 
        << "Alloc, nbytes = " << ByteDisplay(nbytes) << ".";
  
    nbytes = detail::AlignedNBytes<ALIGN_NBYTES>(nbytes);
    MemEntry *free_entry = nullptr;
    // 1. normal case, alloc in train small/large mem pool
    if (nbytes < small_block_nbytes_) {
      free_entry = free_list_small_.PopFreeEntry(nbytes, true, 50);
      CHECK(!alloc_conf::ALWAYS_CHECK_STATE || free_entry == nullptr || CheckState());
    }
    if (free_entry == nullptr && (free_entry = free_list_large_.PopFreeEntry(nbytes, false, 50)) != nullptr) {
      free_entry = Split(free_entry, free_entry->addr_offset, nbytes);
      CHECK(!alloc_conf::ALWAYS_CHECK_STATE || CheckState());
    }
    // 2. find global free phy page to enlarge train memory pool 
    if (free_entry == nullptr && retry_alloc) {
        size_t try_allocate_n = detail::AlignedNBytes<MEM_BLOCK_NBYTES * 8>(nbytes) / MEM_BLOCK_NBYTES;
        std::vector<PhyMem *> phy_mem_list = mempool_.AllocNumPhyMem(policy_, try_allocate_n);
        size_t allocated = phy_mem_list.size();
        LOG_IF(INFO, allocated < try_allocate_n) << log_prefix_ 
            << "Require " << try_allocate_n 
            << " physical memory blocks, but only get " << allocated 
            << ", later retry may fail.";
        if (allocated == 0) {
          DumpState();
          LOG(FATAL) << log_prefix_  << "OOM while finding physical memory page, nbytes = " << ByteDisplay(nbytes) << ".";
        }
        TVMAllocator::Get().SyncAllocTrain(phy_mem_list, lock);
        auto *expand_entry = reinterpret_cast<MemEntry *>(
            MemPool::Get().GetSharedMemory().allocate(sizeof(MemEntry)));
        expand_entry->addr_offset = MEM_BLOCK_NBYTES * mapped_mem_list_.size();
        expand_entry->nbytes = allocated * MEM_BLOCK_NBYTES;
        expand_entry->is_free = false;
        expand_entry->is_small = false;
        expand_entry->is_train = true;
        expand_entry->is_alloc = true;
        expand_entry->rank = 0;
        ExpandMemorySpace(phy_mem_list, allocated);
        entry_list_.LinkNewEntry(expand_entry);
        free_list_large_.PushFreeEntry(expand_entry);
        CHECK(!alloc_conf::ALWAYS_CHECK_STATE || CheckState());
        return Alloc0(nbytes, false, lock);
    }
    if (free_entry == nullptr) {
      DumpState();
      LOG(FATAL) << log_prefix_ << "OMM";
    }
    CHECK(!free_entry->is_free);
    EnsurePhyMemAlloc(free_entry, lock);
    mempool_.AddAllocatedNbytes(free_entry->nbytes, policy_);
    peek_allocated_nbytes_->store(
      std::max(peek_allocated_nbytes_->load(std::memory_order_relaxed), mempool_.GetAllocatedNbytes(policy_)), 
      std::memory_order_relaxed);
    CHECK(!alloc_conf::ALWAYS_CHECK_STATE || CheckState());
    return free_entry;
  }

  void AllocExpandVirtualMemory(size_t nbytes, bip::scoped_lock<bip::interprocess_mutex> &lock) {
    // TVMAllocator::Get().SyncAllocTrain(phy_mem_list, lock);
    // auto *expand_entry = reinterpret_cast<MemEntry *>(
    //     MemPool::Get().GetSharedMemory().allocate(sizeof(MemEntry)));
    // expand_entry->addr_offset = MEM_BLOCK_NBYTES * mapped_mem_list_.size();
    // expand_entry->nbytes = allocated * MEM_BLOCK_NBYTES;
    // expand_entry->is_free = false;
    // expand_entry->is_small = false;
    // expand_entry->is_train = true;
    // expand_entry->is_alloc = true;
    // expand_entry->rank = 0;
    // ExpandMemorySpace(phy_mem_list, allocated);
    // entry_list_.LinkNewEntry(expand_entry);
    // free_list_large_.PushFreeEntry(expand_entry);
    // CHECK(!alloc_conf::ALWAYS_CHECK_STATE || CheckState());
  }

public:
  static TorchAllocator &Get();

  void Free(std::byte *addr) {
    CHECK_GE(addr, base_ptr_);
    CHECK_LT(addr, base_ptr_ + mapped_mem_list_.size() * MEM_BLOCK_NBYTES);
    bip::scoped_lock lock{mempool_.GetMutex()};
    auto *entry = entry_list_.GetEntry(addr - base_ptr_);
    CHECK(entry != nullptr);
    Free0(entry, lock);
  }

  std::byte *Alloc(size_t nbytes) {
    bip::scoped_lock lock{mempool_.GetMutex()};
    return base_ptr_ + Alloc0(nbytes, true, lock)->addr_offset;
  }
  
  void EmptyCache() {
    bip::scoped_lock lock{mempool_.GetMutex()};
    ReleaseFreePhyMem(lock);
    peek_allocated_nbytes_->store(mempool_.GetAllocatedNbytes(policy_), 
                                  std::memory_order_relaxed);
  }

  size_t PeekAllocatedNbytes() {
    return peek_allocated_nbytes_->load(std::memory_order_relaxed);
  }

};

}