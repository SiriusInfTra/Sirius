#include "torch_allocator.h" 
#include "mempool.h"
#include <iostream>
#include <ostream>
#include <tuple>
#include <memory>
#include <functional>
#include <glog/logging.h>

void *col_malloc(ssize_t size, int device, cudaStream_t stream) {
  return colserve::sta::TorchAllocator::Get().Alloc<true>(size);
}
void col_free(void *ptr, ssize_t size, int device, cudaStream_t stream) {
  colserve::sta::TorchAllocator::Get().Free(reinterpret_cast<std::byte*>(ptr));
}

namespace colserve::sta {

void EntryList::LinkNewEntry(TorchMemEntry *entry) {
  if (entry_list_.empty()) {
    CHECK_EQ(entry->addr_offset, 0);
  } else {
    CHECK_EQ(entry->addr_offset, entry_list_.back()->addr_offset + entry_list_.back()->nbytes);
  }

  entry->pos_entrytable =
      entry_by_addr.insert(std::make_pair(entry->addr_offset, entry)).first;
  entry->pos_entrylist = entry_list_.insert(entry_list_.cend(), entry);
}
TorchMemEntry *
EntryList::GetEntry(std::ptrdiff_t addr_offset) {
  auto iter = entry_by_addr.find(addr_offset);
  if (iter == entry_by_addr.cend()) {
    return nullptr;
  }
  auto *entry = iter->second;
  CHECK(iter == entry->pos_entrytable); 
  return entry;
}
TorchMemEntry *
EntryList::GetPrevEntry(TorchMemEntry *entry) {
  auto iter = entry->pos_entrylist;
  if (iter == entry_list_.cbegin()) {
    return nullptr;
  }
  return *std::prev(iter);
}
TorchMemEntry *
EntryList::GetNextEntry(TorchMemEntry *entry) {
  auto iter = std::next(entry->pos_entrylist);
  if (iter == entry_list_.cend()) {
    return nullptr;
  }
  return *iter;
}
TorchMemEntry *
EntryList::SplitFreeEntry(TorchMemEntry *origin_entry,
                                         size_t remain) {
  CHECK_GT(origin_entry->nbytes, remain);
  CHECK(origin_entry->is_free);
  bool insert_success;
  auto *entry_split = new TorchMemEntry;
  /* [origin: remain] [split: nbytes - remain] */
  entry_split->nbytes = origin_entry->nbytes - remain;
  entry_split->is_free = origin_entry->is_free;
  entry_split->addr_offset = origin_entry->addr_offset + remain;
  entry_split->pos_entrylist =
      entry_list_.insert(std::next(origin_entry->pos_entrylist), entry_split);
  entry_split->is_small = origin_entry->is_small;
  origin_entry->nbytes = remain;
  std::tie(entry_split->pos_entrytable, insert_success) = entry_by_addr.insert(
      std::make_pair(entry_split->addr_offset, entry_split));
  CHECK(insert_success);
  CHECK(!torch_allocator::STRICT_CHECK_STATE || CheckState());
  return entry_split;
}
TorchMemEntry *
EntryList::MergeFreeEntry(TorchMemEntry *first_entry,
                                         TorchMemEntry *secound_entry) {
  CHECK_EQ(first_entry->addr_offset + first_entry->nbytes,
           secound_entry->addr_offset);
  CHECK_EQ(GetNextEntry(first_entry), secound_entry);
  CHECK(first_entry->is_free);
  CHECK_EQ(first_entry->is_free, secound_entry->is_free);
  CHECK_EQ(first_entry->is_small, secound_entry->is_small);
  first_entry->nbytes += secound_entry->nbytes;
  entry_list_.erase(secound_entry->pos_entrylist);
  entry_by_addr.erase(secound_entry->pos_entrytable);
  delete secound_entry;
  CHECK(!torch_allocator::STRICT_CHECK_STATE || CheckState());
  return first_entry;
}
void EntryList::DumpTorchMemEntryList(std::ostream &out) {
  out << "start,len,allocated,next,prev,is_free,is_small"
      << "\n";
  for (auto &entry : entry_list_) {
    auto *prev = GetPrevEntry(entry);
    auto *next = GetNextEntry(entry);
    out << entry->addr_offset << "," << entry->nbytes << ","
        << static_cast<int>(entry->is_free) << ","
        << (next ? next->addr_offset : -1) << ","
        << (prev ? prev->addr_offset : -1) << ","
        << static_cast<unsigned>(entry->is_free) << ","
        << static_cast<size_t>(entry->is_small) << "\n";
  }
  out << std::flush;
}
bool EntryList::CheckState() {
  if constexpr (torch_allocator::DUMP_BEFORE_CHECK) {
    LOG(INFO) << "[TorchAllocator] Dump entry_list.";
    DumpTorchMemEntryList(std::cerr);
  }
  for (auto &entry : entry_list_) {
    if (auto *prev = GetPrevEntry(entry); prev) {
      CHECK_EQ(prev->addr_offset + prev->nbytes, entry->addr_offset);
    }
    if (auto *next = GetNextEntry(entry); next) {
      CHECK_EQ(entry->addr_offset + entry->nbytes, next->addr_offset);
    }
  }

  return true;
}

template<bool is_small>
FreeList<is_small>::FreeList(EntryList &list_index)
    : list_index_(list_index) {}

template<bool is_small>
TorchMemEntry *FreeList<is_small>::PopFreeEntry(size_t nbytes) {
  auto iter = entry_by_nbytes.lower_bound(nbytes);
  if (iter == entry_by_nbytes.cend()) {
    return nullptr;
  }
  auto *free_entry = iter->second;
  if (free_entry->nbytes > nbytes ) {
    auto split_entry = list_index_.SplitFreeEntry(free_entry, nbytes);
    split_entry->pos_freelist = entry_by_nbytes.insert(
        std::make_pair(split_entry->nbytes, split_entry));
  }
  entry_by_nbytes.erase(free_entry->pos_freelist);
  free_entry->is_free = false;
  CHECK_EQ(free_entry->is_small, is_small);
  CHECK(!torch_allocator::STRICT_CHECK_STATE || CheckState());
  return free_entry;
}

template<bool is_small>
TorchMemEntry *FreeList<is_small>::PopFreeEntryLarge(size_t nbytes) {
    if (entry_by_nbytes.empty()) { return nullptr; }
    auto largest_entry = std::prev(entry_by_nbytes.cend())->second;
    if (largest_entry->nbytes < nbytes) { return nullptr; }
    entry_by_nbytes.erase(largest_entry->pos_freelist);
    largest_entry->is_free = false;
    CHECK_EQ(largest_entry->is_small, is_small);
    CHECK(!torch_allocator::STRICT_CHECK_STATE || CheckState());
    return largest_entry;
  }

template<bool is_small>
void FreeList<is_small>::PushFreeEntry(TorchMemEntry *entry) {
  CHECK_EQ(entry->is_free, false);
  CHECK_EQ(entry->is_small, is_small);
  entry->is_free = true;
  entry->is_small = is_small;

  if (auto prev_entry = list_index_.GetPrevEntry(entry);
      prev_entry && prev_entry->is_free &&
      prev_entry->is_small == entry->is_small) {
    entry_by_nbytes.erase(prev_entry->pos_freelist);
    entry = list_index_.MergeFreeEntry(prev_entry, entry);
  }
  if (auto next_entry = list_index_.GetNextEntry(entry);
      next_entry && next_entry->is_free &&
      next_entry->is_small == entry->is_small) {
    entry_by_nbytes.erase(next_entry->pos_freelist);
    entry = list_index_.MergeFreeEntry(entry, next_entry);
  }

  entry->pos_freelist =
      entry_by_nbytes.insert(std::make_pair(entry->nbytes, entry));
  
  CHECK(!torch_allocator::STRICT_CHECK_STATE || CheckState());
}

template<bool is_small>
void FreeList<is_small>::DumpFreeList(std::ostream &out) {
  out << "start,len,next,prev,is_free,is_small"
      << "\n";
  for (auto &&[nbytes, entry] : entry_by_nbytes) {
    auto *prev = list_index_.GetPrevEntry(entry);
    auto *next = list_index_.GetNextEntry(entry);
    out << entry->addr_offset << "," << entry->nbytes << ","
        << (next ? next->addr_offset : -1) << ","
        << (prev ? prev->addr_offset : -1) << ","
        << static_cast<unsigned>(entry->is_free) << ","
        << static_cast<size_t>(entry->is_small) << "\n";
  }
  out << std::flush;
}

template<bool is_small>
bool FreeList<is_small>::CheckState() {
  if constexpr (torch_allocator::DUMP_BEFORE_CHECK) {
    LOG(INFO) << "[TorchAllocator] Dump free_list.";
    DumpFreeList(std::cerr);
  }
  for (auto &&[nbytes, entry] : entry_by_nbytes) {
    CHECK_EQ(entry->is_free, true);
    CHECK_EQ(entry->nbytes, nbytes);
    CHECK_EQ(entry->is_small, is_small);
  }
  return true;
}

template class FreeList<pool::Small>;
template class FreeList<pool::Large>;

std::unique_ptr<TorchAllocator> TorchAllocator::instance_ = nullptr;


TorchAllocator &TorchAllocator::Get() {
  if (instance_ == nullptr) {
    instance_.reset(new TorchAllocator(MemPool::Get()));
  }
  return *instance_;
}

TorchAllocator::TorchAllocator(MemPool &mempool)
    : mempool_(mempool), free_list_small_(entry_list_),
      free_list_large_(entry_list_) {
  CU_CALL(cuMemAddressReserve(reinterpret_cast<CUdeviceptr *>(&base_ptr_),
                              mempool_.mempool_nbytes, MEM_BLOCK_NBYTES, 0, 0));
  LOG(INFO) << "[TorchAllocator] Init torch allocator, dev_ptr = " << base_ptr_ << ".";
}
TorchAllocator::~TorchAllocator() {
  int ignore;
  if (cuDriverGetVersion(&ignore) != CUDA_ERROR_DEINITIALIZED) {
    CU_CALL(cuMemAddressFree(reinterpret_cast<CUdeviceptr>(base_ptr_), mempool_.mempool_nbytes));
  }

}

void TorchAllocator::ExpandMemorySpace(
    const std::vector<PhyMem *> &phy_mem_list, size_t len) {
  auto *mapping_begin = base_ptr_ + mapped_mem_list_.size() * MEM_BLOCK_NBYTES;
  for (size_t k = 0; k < len; ++k) {
    auto *mem_ptr = phy_mem_list[k];
    CU_CALL(
        cuMemMap(reinterpret_cast<CUdeviceptr>(
                     base_ptr_ + mapped_mem_list_.size() * MEM_BLOCK_NBYTES),
                 MEM_BLOCK_NBYTES, 0, mem_ptr->cu_handle, 0));
    mapped_mem_list_.push_back(mem_ptr);
  }
  CUmemAccessDesc acc_desc = {
      .location = {.type = CU_MEM_LOCATION_TYPE_DEVICE, .id = 0},
      .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE};
  CU_CALL(cuMemSetAccess(reinterpret_cast<CUdeviceptr>(mapping_begin),
                         phy_mem_list.size() * MEM_BLOCK_NBYTES, &acc_desc, 1));
}
bool TorchAllocator::CheckState() {
  entry_list_.CheckState();
  free_list_small_.CheckState();
  free_list_large_.CheckState();
  return true;
}

template<bool retry_alloc>
std::byte *TorchAllocator::Alloc(size_t unaligned_nbytes) {
  if (unaligned_nbytes == 0) { return nullptr; }
  auto &&[is_small, aligned_nbytes] = detail::AlignNbytes(unaligned_nbytes);
  LOG_IF(INFO, torch_allocator::VERBOSE) << "[TorchAllocator] Alloc " << detail::ByteDisplay(unaligned_nbytes) << ", (aligned to " << detail::ByteDisplay(aligned_nbytes) << ").";
  TorchMemEntry *free_entry = nullptr;
  if (is_small) {
    free_entry = free_list_small_.PopFreeEntry(aligned_nbytes);
    if (free_entry == nullptr) {
      /* try to pop entry from large */
      auto *free_entry_large =
          free_list_large_.PopFreeEntry(detail::SMALL_PAGE_NBYTES);
      if (free_entry_large != nullptr) {
        free_entry_large->is_small = true;
        free_list_small_.PushFreeEntry(free_entry_large);
        free_entry = free_list_small_.PopFreeEntry(aligned_nbytes);
      }
    }
  } else {
    free_entry = free_list_large_.PopFreeEntry(aligned_nbytes);
  }
  if constexpr (retry_alloc) {
    if (free_entry == nullptr) {
      /* try to allocate from global mempool*/
      std::vector<PhyMem *> phy_mem_list{
          detail::AlignedNBytes<MEM_BLOCK_NBYTES * 8>(aligned_nbytes) /
          MEM_BLOCK_NBYTES};
      size_t allocated = mempool_.AllocPhyMem(phy_mem_list, Belong::kTrain);
      LOG_IF(INFO, allocated < phy_mem_list.size())
          << "[Torch Allocator] Require " << phy_mem_list.size()
          << " physical memory blocks, but only get " << allocated << ", later retry may fail.";
      CHECK_GT(allocated, 0);
      auto *expand_entry = new TorchMemEntry;
      expand_entry->addr_offset = MEM_BLOCK_NBYTES * mapped_mem_list_.size();
      expand_entry->nbytes = allocated * MEM_BLOCK_NBYTES;
      expand_entry->is_free = false;
      expand_entry->is_small = false;
      ExpandMemorySpace(phy_mem_list, allocated);
      entry_list_.LinkNewEntry(expand_entry);
      free_list_large_.PushFreeEntry(expand_entry);
      CHECK(!torch_allocator::ALWAYS_CHECK_STATE || CheckState());
      return Alloc<false>(unaligned_nbytes);
    }
  }
  CHECK(!torch_allocator::ALWAYS_CHECK_STATE || CheckState());
  EnsurePhyMemAlloc(free_entry);
  return base_ptr_ + free_entry->addr_offset;
}

template std::byte *TorchAllocator::Alloc<true>(size_t unaligned_nbytes);
template std::byte *TorchAllocator::Alloc<false>(size_t unaligned_nbytes);


void TorchAllocator::Free(std::byte *ptr) {
  if (ptr == nullptr) { return; }
  CHECK_GE(ptr, base_ptr_);
  CHECK_LT(ptr, base_ptr_ + mapped_mem_list_.size() * MEM_BLOCK_NBYTES);
  auto addr_offset = ptr - base_ptr_;
  auto entry = entry_list_.GetEntry(addr_offset);
  LOG_IF(INFO, torch_allocator::VERBOSE) << "[TorchAllocator] Free, nbytes = " << entry->nbytes << ".";
  CHECK(entry != nullptr);
  CHECK(!entry->is_free);
  if (entry->is_small) {
    free_list_small_.PushFreeEntry(entry);
    while(auto large_entry = free_list_small_.PopFreeEntryLarge(detail::MIN_BLOCK_NBYTES)) {
      large_entry->is_small = false;
      free_list_large_.PushFreeEntry(large_entry);
    }
  } else {
    free_list_large_.PushFreeEntry(entry);
  }
  CHECK(!torch_allocator::ALWAYS_CHECK_STATE || CheckState());
}


} // namespace colserve::sta
