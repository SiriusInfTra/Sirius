#pragma once

#include <common/generic_allocator.h>
#include <common/mempool.h>
#include <common/util.h>

#include <glog/logging.h>
#include <algorithm>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/lock_options.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <memory>
#include <mutex>
#include <numeric>
#include <sys/types.h>
#include <map>
#include <ostream>
#include <list>
#include <random>
#include <sstream>
#include <utility>
#include <vector>


namespace colserve::sta {

class TVMAllocator : public GenericAllocator {
public:
  static const constexpr size_t ALIGN_NBYTES = 16_MB;
  static const constexpr size_t DUPLICATE_SHUFFLE_K = 1;
  static_assert(VA_RESERVE_SCALE >= DUPLICATE_SHUFFLE_K, "Must reserve enough virtual memory for duplicate shuffle.");

  static size_t AlignNBytes(size_t nbytes) {
    return nbytes >= MEM_BLOCK_NBYTES ? detail::AlignedNBytes<MEM_BLOCK_NBYTES>(nbytes) : detail::AlignedNBytes<ALIGN_NBYTES>(nbytes);
  }

private:
  static std::unique_ptr<TVMAllocator> instance_;
  std::mutex mutex_;
  std::vector<std::vector<size_t>> duplicate_shuffle_map_;

std::vector<size_t> Shuffle(const std::vector<size_t> &arr, size_t k, std::mt19937 &rng) {
    CHECK_EQ(arr.size() % k, 0) << arr.size();
    std::vector<size_t> index(arr.size() / k);
    std::iota(index.begin(), index.end(), 0);
    std::shuffle(index.begin(), index.end(), rng);
    std::vector<size_t> shuffledArr(arr.size());
    for (size_t i = 0; i < arr.size(); ++i) {
        shuffledArr[i] = arr[index[i / k] * k  + i % k];
    }
    return shuffledArr;
}

  void InitDuplicate() {
    size_t phy_num = mempool_.GetPhyMemList().size(); // number of physical memory pages
    size_t vir_num = phy_num * DUPLICATE_SHUFFLE_K;   // number of virtual memory pages
    size_t vir_num_per = DUPLICATE_SHUFFLE_K;         // number of virtual memory pages per physical memory pages
    duplicate_shuffle_map_ = std::vector<std::vector<size_t>>(vir_num, std::vector<size_t>(vir_num_per));
    // init;
    std::mt19937 rng{42};
    for (size_t k = 0; k < vir_num_per; ++k) {
      std::vector<size_t> map_to(phy_num);
      std::iota(map_to.begin(), map_to.end(), k * phy_num);
      if (k != 0) { 
        map_to = Shuffle(map_to, 4, rng);
      }
      for (size_t i = 0; i < phy_num; ++i) { 
        duplicate_shuffle_map_[i][k] = map_to[i]; 
      }
    }
    for (size_t k = 1; k < vir_num_per; ++k) {
      for (size_t i = 0; i < phy_num; ++i) {
        size_t j = duplicate_shuffle_map_[i][k];
        duplicate_shuffle_map_[j] = duplicate_shuffle_map_[i];
      }
    }
    LOG(INFO) << log_prefix_ << "Duplicate entry map inited.";
    for (size_t i = 0; i < vir_num; ++i) {
      std::stringstream ss;
      for (size_t j = 0; j < vir_num_per; ++j) {
        ss << duplicate_shuffle_map_[i][j] << " ";
      }
      DLOG(INFO) << log_prefix_ << ss.str();
    }
  }

  void AllocDuplicateEntry(MemEntry *entry) {
    CHECK_EQ(entry->addr_offset % MEM_BLOCK_NBYTES, 0) << entry;
    CHECK_EQ(entry->nbytes % MEM_BLOCK_NBYTES, 0) << entry;
    auto &&[index_begin, index_end] = GetAssociatedPhyMemIndex(entry);
    for (size_t k = index_begin; k < index_end; ++k) {
      for (size_t i : duplicate_shuffle_map_[k]) {
        if (k == i) { continue; }
        auto *entry1 = entry_list_.GetEntryWithAddr(i * MEM_BLOCK_NBYTES);
        CHECK(entry1 != nullptr) << "Unable to find entry1 with addr_offset = " << i * MEM_BLOCK_NBYTES << ".";
        CHECK_LE(entry1->addr_offset, i * MEM_BLOCK_NBYTES) << entry1 << entry;
        CHECK_GT(entry1->addr_offset + entry1->nbytes, i * MEM_BLOCK_NBYTES) << entry1 << entry;
        CHECK_EQ(entry1->nbytes % MEM_BLOCK_NBYTES, 0) << entry1 << entry;
        CHECK_NE(entry1->rank, entry->rank) << entry1 << entry;
        CHECK(entry1->is_free) << entry1 << entry;
        CHECK(!entry1->is_small) << entry1 << entry;
        free_list_large_.PopFreeEntry(entry1);
        entry1 = Split(entry1, i * MEM_BLOCK_NBYTES, MEM_BLOCK_NBYTES);
        entry1->is_train = entry->is_train;
      }
    }
  }

  void ReleaseDuplicateEntry(MemEntry *entry) {
    CHECK_EQ(entry->addr_offset % MEM_BLOCK_NBYTES, 0);
    CHECK_EQ(entry->nbytes % MEM_BLOCK_NBYTES, 0);
    auto &&[index_begin, index_end] = GetAssociatedPhyMemIndex(entry);
    
    for (size_t k = index_begin; k < index_end; ++k) {
      for (size_t i : duplicate_shuffle_map_[k]) {
        if (k == i) { continue; }
        auto *entry1 = entry_list_.GetEntry(i * MEM_BLOCK_NBYTES);
        CHECK(entry1 != nullptr);
        CHECK_EQ(entry1->addr_offset, i * MEM_BLOCK_NBYTES);
        CHECK_EQ(entry1->nbytes, MEM_BLOCK_NBYTES);
        CHECK_NE(entry1->rank, entry->rank) << entry1 << entry;
        CHECK(!entry1->is_free);
        CHECK(!entry1->is_small);
        entry1->is_train = false;
        free_list_large_.PushFreeEntry(entry1);
      }
    }
  }

  MemEntry *Alloc(size_t nbytes, bip::scoped_lock<bip::interprocess_mutex> &lock) {
    CHECK(lock.owns());
    CHECK_GT(nbytes, 0);
    LOG_IF(INFO, alloc_conf::VERBOSE) << log_prefix_ << "Alloc, nbytes = " << ByteDisplay(nbytes) << ".";
    nbytes = AlignNBytes(nbytes);
    MemEntry *free_entry = nullptr;
    if (nbytes < small_block_nbytes_) {
      free_entry = free_list_small_.PopFreeEntry(nbytes, true);
      if (free_entry == nullptr && (free_entry = free_list_large_.PopFreeEntry(MEM_BLOCK_NBYTES, true)) != nullptr) {
        AllocDuplicateEntry(free_entry);
        free_entry->is_small = true;
        CHECK_EQ(free_entry->nbytes % MEM_BLOCK_NBYTES, 0);
        free_entry = Split(free_entry, free_entry->addr_offset, nbytes);
      }
      CHECK(!alloc_conf::ALWAYS_CHECK_STATE || free_entry == nullptr || CheckState());
    } else {
      free_entry = free_list_large_.PopFreeEntry(nbytes, true);
      if (free_entry != nullptr) {
        AllocDuplicateEntry(free_entry);
      }
    }

    if (free_entry == nullptr) {
      DumpState();
      LOG(FATAL) << log_prefix_  << "OOM while finding virtual memory, nbytes = " << ByteDisplay(nbytes) << ".";
    }
    auto &&[index_begin, index_end] = GetAssociatedPhyMemIndex(free_entry);
    std::vector<PhyMem *> phy_mem_list;
    for (size_t k = index_begin; k < index_end; ++k) {
      if (*mapped_mem_list_[k]->belong == Belong::kFree) {
        phy_mem_list.push_back(mapped_mem_list_[k]);
      }
    }

    // if (free_entry->is_small) {
    //   CHECK_EQ(free_entry->addr_offset / MEM_BLOCK_NBYTES, (free_entry->addr_offset + free_entry->nbytes - 1) / MEM_BLOCK_NBYTES);
    //   free_entry = free_list_small_.PopFreeEntry(free_entry);
    // } else {
    //   for (size_t k = index_begin; k < index_end; ++k) {

    //   }
    // }

    mempool_.AllocSpecifiedPhyMem(phy_mem_list, policy_);
    mempool_.AddAllocatedNbytes(nbytes, policy_);
    CHECK(!alloc_conf::ALWAYS_CHECK_STATE || CheckState());
    return free_entry;
  }

  MemEntry *Free(MemEntry *entry, bip::scoped_lock<bip::interprocess_mutex> &lock) {
    CHECK(lock.owns());
    CHECK(entry != nullptr);
    CHECK(!entry->is_free);
    size_t nbytes = entry->nbytes;
    LOG_IF(INFO, alloc_conf::VERBOSE) << log_prefix_ 
        << "Free, nbytes = " << ByteDisplay(entry->nbytes) 
        << "ptr = " << base_ptr_ + entry->addr_offset << ".";
    
    // 1. maintain phy memory info
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


    // 2. free list
    if (entry->is_small) {
      entry = free_list_small_.PushFreeEntry(entry);
      if (entry->nbytes == MEM_BLOCK_NBYTES) {
        entry = free_list_small_.PopFreeEntry(entry);
        ReleaseDuplicateEntry(entry);
        entry->is_small = false;
        free_list_large_.PushFreeEntry(entry);
      }
    } else { // large
      // 2.0. merge with large
      ReleaseDuplicateEntry(entry);
      entry = free_list_large_.PushFreeEntry(entry);
    }
    CHECK(!alloc_conf::ALWAYS_CHECK_STATE || CheckState());
    return entry;
  }

public:
  static TVMAllocator &Get();

  static void Init(bool for_train, bip::scoped_lock<bip::interprocess_mutex> &lock);

  TVMAllocator(MemPool &mempool, bool for_train, bip::scoped_lock<bip::interprocess_mutex> &lock);

  // TO FIX: Sync is not good name
  void SyncAllocTrain_NOT_USE(const std::vector<PhyMem *> &phymem_list, bip::scoped_lock<bip::interprocess_mutex> &lock) {
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
        CHECK(entry->is_free) << entry;

        if (entry->is_small) {
          DumpState();
          CHECK(!entry->is_small) << entry;
        }
        entry = free_list_large_.PopFreeEntry(entry);
        entry = Split(entry, begin, end - begin);
        entry->is_train = true;
        AllocDuplicateEntry(entry);
        return entry;
      });
    }
  }

  std::vector<PhyMem *> AllocForTrain(size_t n, bip::scoped_lock<bip::interprocess_mutex> &lock) {
    CHECK(lock.owns());
    LOG_IF(INFO, alloc_conf::VERBOSE) << log_prefix_ << "Alloc for train "  << n << ".";
    std::vector<PhyMem *> phy_mem_list;
    for (size_t k = 0; k < n; ++k) {
      auto *free_entry = free_list_large_.PopFreeEntry(MEM_BLOCK_NBYTES, true);
      if (free_entry == nullptr) { break; }
      free_entry->is_train = true;
      CHECK_EQ(free_entry->addr_offset % MEM_BLOCK_NBYTES, 0);
      size_t page_index = duplicate_shuffle_map_[free_entry->addr_offset / MEM_BLOCK_NBYTES][0];
      phy_mem_list.push_back(&mempool_.GetPhyMemList()[page_index]);
      AllocDuplicateEntry(free_entry);
    }
    mempool_.AllocSpecifiedPhyMem(phy_mem_list, Belong::kTrain);
    mempool_.AddAllocatedNbytes(n * MEM_BLOCK_NBYTES, Belong::kTrain);
    CHECK(!alloc_conf::ALWAYS_CHECK_STATE || CheckState());
    return phy_mem_list;
  }

  void FreeForTrain(const std::vector<PhyMem *> &phy_mem_list, bip::scoped_lock<bip::interprocess_mutex> &lock) {
    CHECK(lock.owns());
    LOG_IF(INFO, alloc_conf::VERBOSE) << log_prefix_ << "Free for train " << phy_mem_list.size() << ".";
    for (auto phy_mem : phy_mem_list) {
      CHECK_EQ(*phy_mem->belong, Belong::kTrain);
      auto *entry = entry_list_.GetEntry(phy_mem->index * MEM_BLOCK_NBYTES);
      CHECK(entry != nullptr);
      CHECK(entry->is_train) << entry;
      CHECK_EQ(entry->nbytes, MEM_BLOCK_NBYTES);
      Free(entry, lock);
    }
    mempool_.SubAllocatedNbytes(phy_mem_list.size() * MEM_BLOCK_NBYTES, Belong::kTrain);
    mempool_.DeallocPhyMem(phy_mem_list);
  }

  void SyncFreeTrain_NOT_USE(const std::vector<PhyMem *> &phymem_list, bip::scoped_lock<bip::interprocess_mutex> &lock) {
    CHECK(lock.owns());
    LOG_IF(INFO, alloc_conf::VERBOSE) << log_prefix_ << "SyncFreeTrain: " << phymem_list.size();
    for (auto *phymem : phymem_list) {
      ptrdiff_t addr_offset = phymem->index * MEM_BLOCK_NBYTES;
      size_t nbytes = MEM_BLOCK_NBYTES;
      entry_list_.IterateRange(addr_offset, nbytes, [&](MemEntry *entry) {
        CHECK_EQ(entry->addr_offset, addr_offset) << entry;
        CHECK_EQ(entry->nbytes, nbytes) << entry;
        CHECK(entry->is_train) << entry;
        CHECK(!entry->is_free) << entry;
        entry = Free(entry, lock);
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
    Free(entry, lock);
  }

  std::byte *Alloc(size_t nbytes) {
    bip::scoped_lock lock{mempool_.GetMutex()};
    return base_ptr_ + Alloc(nbytes, lock)->addr_offset;
  }
};

  
}