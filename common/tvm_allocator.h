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
#include <numeric>
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
public:
  static const constexpr size_t ALIGN_NBYTES = 16_MB; 
private:
  static std::unique_ptr<TVMAllocator> instance_;
  std::mutex mutex_;

  MemEntry *Alloc0(size_t nbytes) {
    CHECK_GT(nbytes, 0);
    LOG_IF(INFO, alloc_conf::VERBOSE) << log_prefix_ << "Alloc, nbytes = " << detail::ByteDisplay(nbytes) << ".";
  
    nbytes = detail::AlignedNBytes<ALIGN_NBYTES>(nbytes);
    MemEntry *free_entry = nullptr;
    if (nbytes < SMALL_BLOCK_NBYTES) {
      free_entry = free_list_small_.PopFreeEntry(nbytes, true);
      CHECK(!alloc_conf::ALWAYS_CHECK_STATE || free_entry == nullptr || CheckState());
    }
    if (free_entry == nullptr && (free_entry = free_list_large_.PopFreeEntry(nbytes, false)) != nullptr) {
      free_entry = Split(free_entry, free_entry->addr_offset + free_entry->nbytes - nbytes, nbytes);
      CHECK(!alloc_conf::ALWAYS_CHECK_STATE || CheckState());
    }

    if (free_entry == nullptr) {
      DumpState();
      LOG(FATAL) << log_prefix_ << "OOM";
    }
    auto &&[index_begin, index_end] = GetAssociatedPhyMemIndex(free_entry);
    std::vector<PhyMem *> phy_mem_list;
    for (size_t k = index_begin; k < index_end; ++k) {
      if (*mapped_mem_list_[k]->belong == Belong::kFree) {
        phy_mem_list.push_back(mapped_mem_list_[k]);
      }
    }
    mempool_.ClaimPhyMem(phy_mem_list, policy_);
    mempool_.AddAllocatedNbytes(nbytes, policy_);
    CHECK(!alloc_conf::ALWAYS_CHECK_STATE || CheckState());
    return free_entry;
  }

  MemEntry *Free0(MemEntry *entry) {
    CHECK(entry != nullptr);
    CHECK(!entry->is_free);
    size_t nbytes = entry->nbytes;
    LOG_IF(INFO, alloc_conf::VERBOSE) << log_prefix_ << "Free, nbytes = " << detail::ByteDisplay(entry->nbytes) << "ptr = " << base_ptr_ + entry->addr_offset << ".";
    if (entry->is_train) {
      entry->is_train = false;
    } else {
      mempool_.SubAllocatedNbytes(entry->nbytes, policy_);
      auto &&[index_begin, index_end] = GetAssociatedPhyMemIndex(entry);
      std::vector<PhyMem *> phy_mem_list; 
      for (size_t k = index_begin; k < index_end; ++k) {
        bool page_free = true;
        if (k == index_begin) {
          auto *maybe_free_entry = entry;
          while (true) {
            //  && !maybe_free_entry->is_train
            if (maybe_free_entry != entry && !maybe_free_entry->is_free) {
              CHECK(!maybe_free_entry->is_train);
              page_free = false;
              break;
            }
            if (maybe_free_entry->addr_offset <= index_begin * MEM_BLOCK_NBYTES) {
              break;
            }
            maybe_free_entry = entry_list_.GetPrevEntry(maybe_free_entry);
          }
        }
        if (page_free && k == index_end - 1) {
          auto *maybe_free_entry = entry;
          while (true) {
            //  && !maybe_free_entry->is_train
            if (maybe_free_entry != entry && !maybe_free_entry->is_free) {
              CHECK(!maybe_free_entry->is_train);
              page_free = false;
              break;
            }
            if (maybe_free_entry->addr_offset + maybe_free_entry->nbytes >= index_end * MEM_BLOCK_NBYTES) {
              break;
            }
            maybe_free_entry = entry_list_.GetNextEntry(maybe_free_entry);
          }
        }
        if (page_free) {
          phy_mem_list.push_back(mapped_mem_list_[k]);
        }
      }
      mempool_.DeallocPhyMem(phy_mem_list);
    }
    if (entry->is_small) {
      entry = free_list_small_.PushFreeEntry(entry);
      entry = MaybeMerge(entry);
    } else {
      entry = free_list_large_.PushFreeEntry(entry);
      if (auto *prev_entry_nbytes = entry_list_.GetPrevEntry(entry); prev_entry_nbytes && prev_entry_nbytes->is_small && prev_entry_nbytes->is_free) {
        size_t prev_nbytes = prev_entry_nbytes->nbytes;
        if (auto *maybe_merged_entry = MaybeMerge(prev_entry_nbytes); maybe_merged_entry->nbytes > prev_nbytes) {
          entry = maybe_merged_entry;
        }
      }
      if (auto *next_entry = entry_list_.GetNextEntry(entry); next_entry && next_entry->is_small && next_entry->is_free) {
        size_t next_entry_nbytes = next_entry->nbytes;
        if (auto *maybe_merged_entry = MaybeMerge(next_entry); maybe_merged_entry->nbytes > next_entry_nbytes) {
          entry = maybe_merged_entry;
        }
      }
    }
    CHECK(!alloc_conf::ALWAYS_CHECK_STATE || CheckState());
    return entry;
  }
public:
  static TVMAllocator &Get();

  static void Init(bool for_train, bip::scoped_lock<bip::interprocess_mutex> &lock);

  TVMAllocator(MemPool &mempool, bool for_train, bip::scoped_lock<bip::interprocess_mutex> &lock);

  void SyncAllocTrain(const std::vector<PhyMem *> &phymem_list, bip::scoped_lock<bip::interprocess_mutex> &lock) {
    // TODO: awareness optimal 
    CHECK(lock.owns());
    LOG_IF(INFO, alloc_conf::VERBOSE) << log_prefix_ << "SyncAllocTrain: " << phymem_list.size();
    for (auto *phymem : phymem_list) {
      ptrdiff_t addr_offset = phymem->index * MEM_BLOCK_NBYTES;
      size_t nbytes = MEM_BLOCK_NBYTES;
      entry_list_.IterateRange(addr_offset, nbytes, [&](MemEntry *entry) {
        ptrdiff_t begin = std::max(entry->addr_offset, addr_offset);
        size_t end = std::min(entry->addr_offset + entry->nbytes, addr_offset + nbytes);
        CHECK_GT(end, begin);
        CHECK(entry->is_free);
        entry = (entry->is_small ? free_list_small_ : free_list_large_).PopFreeEntry(entry);
        entry = Split(entry, begin, end - begin);
        entry->is_train = true;
        return entry;
      });
    }
  }

  void SyncFreeTrain(const std::vector<PhyMem *> &phymem_list, bip::scoped_lock<bip::interprocess_mutex> &lock) {
    CHECK(lock.owns());
    LOG_IF(INFO, alloc_conf::VERBOSE) << log_prefix_ << "SyncFreeTrain: " << phymem_list.size();
    for (auto *phymem : phymem_list) {
      ptrdiff_t addr_offset = phymem->index * MEM_BLOCK_NBYTES;
      size_t nbytes = MEM_BLOCK_NBYTES;
      entry_list_.IterateRange(addr_offset, nbytes, [&](MemEntry *entry) {
        CHECK_GE(entry->addr_offset, addr_offset);
        CHECK_LE(entry->addr_offset + entry->nbytes, addr_offset + nbytes);
        CHECK(entry->is_train) << entry;
        CHECK(!entry->is_free) << entry;
        entry = Free0(entry);
        return entry;
      });
    }
  }

  void Free(std::byte *addr) {
    CHECK_GE(addr, base_ptr_);
    CHECK_LT(addr, base_ptr_ + mapped_mem_list_.size() * MEM_BLOCK_NBYTES);
    bip::scoped_lock lock{mempool_.GetMutex()};
    auto *entry = entry_list_.GetEntry(addr - base_ptr_);
    CHECK(entry != nullptr);
    Free0(entry);
  }

  

  std::byte *Alloc(size_t nbytes) {
    bip::scoped_lock lock{mempool_.GetMutex()};
    return base_ptr_ + Alloc0(nbytes)->addr_offset;
  }

  void EmptyCache() {}
};

  
}