#include "generic_allocator.h"
#include "mempool.h"
#include <glog/logging.h>


namespace colserve::sta {
void EntryList::LinkNewEntry(MemEntry *entry) {
  if (entry_list_.empty()) {
    CHECK_EQ(entry->addr_offset, 0);
  } else {
    CHECK_EQ(entry->addr_offset, entry_list_.back()->addr_offset + entry_list_.back()->nbytes);
  }

  entry->pos_entrytable =
      entry_by_addr.insert(std::make_pair(entry->addr_offset, entry)).first;
  entry->pos_entrylist = entry_list_.insert(entry_list_.cend(), entry);
}
MemEntry *
EntryList::GetEntry(std::ptrdiff_t addr_offset) {
  auto iter = entry_by_addr.find(addr_offset);
  if (iter == entry_by_addr.cend()) {
    return nullptr;
  }
  auto *entry = iter->second;
  CHECK(iter == entry->pos_entrytable); 
  return entry;
}
MemEntry *
EntryList::GetPrevEntry(MemEntry *entry) {
  auto iter = entry->pos_entrylist;
  if (iter == entry_list_.cbegin()) {
    return nullptr;
  }
  return *std::prev(iter);
}
MemEntry *
EntryList::GetNextEntry(MemEntry *entry) {
  auto iter = std::next(entry->pos_entrylist);
  if (iter == entry_list_.cend()) {
    return nullptr;
  }
  return *iter;
}
MemEntry *
EntryList::SplitEntry(MemEntry *origin_entry,
                                         size_t remain) {
  CHECK_GT(origin_entry->nbytes, remain);
  bool insert_success;
  auto *entry_split = new MemEntry;
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
MemEntry *
EntryList::MergeMemEntry(MemEntry *first_entry,
                                         MemEntry *secound_entry) {
  CHECK_EQ(first_entry->addr_offset + first_entry->nbytes,
           secound_entry->addr_offset);
  CHECK_EQ(GetNextEntry(first_entry), secound_entry);
  CHECK_EQ(first_entry->is_free, secound_entry->is_free);
  CHECK_EQ(first_entry->is_small, secound_entry->is_small);
  first_entry->nbytes += secound_entry->nbytes;
  entry_list_.erase(secound_entry->pos_entrylist);
  entry_by_addr.erase(secound_entry->pos_entrytable);
  delete secound_entry;
  CHECK(!torch_allocator::STRICT_CHECK_STATE || CheckState());
  return first_entry;
}
void EntryList::DumpMemEntryList(std::ostream &out) {
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
    DumpMemEntryList(std::cerr);
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


FreeList::FreeList(EntryList &list_index, bool is_small, Belong policy)
    : list_index_(list_index), is_small_(is_small), policy_(policy) {}


MemEntry *FreeList::PopFreeEntry(size_t nbytes) {
  auto iter = entry_by_nbytes.lower_bound(nbytes);
  if (iter == entry_by_nbytes.cend()) {
    return nullptr;
  }
  auto *free_entry = iter->second;
  if (free_entry->nbytes > nbytes ) {
    auto split_entry = list_index_.SplitEntry(free_entry, nbytes);
    split_entry->pos_freelist = entry_by_nbytes.insert(
        std::make_pair(split_entry->nbytes, split_entry));
  }
  entry_by_nbytes.erase(free_entry->pos_freelist);
  free_entry->is_free = false;
  CHECK_EQ(free_entry->is_small, is_small_);
  CHECK(!torch_allocator::STRICT_CHECK_STATE || CheckState());
  return free_entry;
}


MemEntry *FreeList::PopFreeEntry(MemEntry *free_entry) {
  CHECK_EQ(free_entry->is_small, is_small_);
  CHECK(free_entry->is_free);
  entry_by_nbytes.erase(free_entry->pos_freelist);
  free_entry->is_free = false;
  CHECK_EQ(free_entry->is_small, is_small_);
  CHECK(!torch_allocator::STRICT_CHECK_STATE || CheckState());
  return free_entry;
}


MemEntry *FreeList::PopFreeEntryLarge(size_t nbytes) {
    if (entry_by_nbytes.empty()) { return nullptr; }
    auto largest_entry = std::prev(entry_by_nbytes.cend())->second;
    if (largest_entry->nbytes < nbytes) { return nullptr; }
    entry_by_nbytes.erase(largest_entry->pos_freelist);
    largest_entry->is_free = false;
    CHECK_EQ(largest_entry->is_small, is_small_);
    CHECK(!torch_allocator::STRICT_CHECK_STATE || CheckState());
    return largest_entry;
  }


MemEntry* FreeList::PushFreeEntry(MemEntry *entry) {
  CHECK_EQ(entry->is_free, false);
  CHECK_EQ(entry->is_small, is_small_);
  entry->is_free = true;
  entry->is_small = is_small_;
  // if (policy_ == Belong::kInfer && !is_small_) {
  //   if (!is_small_) {
  //     // CHECK((entry->addr_offset % MEM_BLOCK_NBYTES != 0) || entry->nbytes < MEM_BLOCK_NBYTES) << entry->addr_offset << " " << entry->nbytes;
  //   } else {
  //   }
  // }

  if (auto prev_entry = list_index_.GetPrevEntry(entry);
      prev_entry && prev_entry->is_free &&
      prev_entry->is_small == entry->is_small) {
    entry_by_nbytes.erase(prev_entry->pos_freelist);
    entry = list_index_.MergeMemEntry(prev_entry, entry);
  }
  if (auto next_entry = list_index_.GetNextEntry(entry);
      next_entry && next_entry->is_free &&
      next_entry->is_small == entry->is_small) {
    entry_by_nbytes.erase(next_entry->pos_freelist);
    entry = list_index_.MergeMemEntry(entry, next_entry);
  }

  entry->pos_freelist =
      entry_by_nbytes.insert(std::make_pair(entry->nbytes, entry));
  if (entry->nbytes > 2_GB && is_small_ == true) {
    LOG(INFO) << "test";
    LOG(FATAL) << "should crash";
  }
  CHECK(!torch_allocator::STRICT_CHECK_STATE || CheckState());
  return entry;
}


void FreeList::DumpFreeList(std::ostream &out) {
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


bool FreeList::CheckState() {
  if constexpr (torch_allocator::DUMP_BEFORE_CHECK) {
    LOG(INFO) << "[TorchAllocator] Dump free_list.";
    DumpFreeList(std::cerr);
  }
  for (auto &&[nbytes, entry] : entry_by_nbytes) {
    CHECK_EQ(entry->is_free, true);
    CHECK_EQ(entry->nbytes, nbytes);
    CHECK_EQ(entry->is_small, is_small_);
  }
  return true;
}


GenericAllocator::GenericAllocator(MemPool &mempool, Belong policy)
    : mempool_(mempool), free_list_small_(entry_list_, true, policy),
      free_list_large_(entry_list_, false, policy), cached_nbytes_(0), allocated_nbytes_(0), policy_(policy) {
  CU_CALL(cuMemAddressReserve(reinterpret_cast<CUdeviceptr *>(&base_ptr_),
                              mempool_.mempool_nbytes * VA_RESERVE_SCALE, MEM_BLOCK_NBYTES, 0, 0));
  LOG(INFO) << "[TorchAllocator] Init torch allocator, dev_ptr = " << base_ptr_ << ".";
}


GenericAllocator::~GenericAllocator() {
  int ignore;
  if (cuDriverGetVersion(&ignore) != CUDA_ERROR_DEINITIALIZED) {
    CU_CALL(cuMemAddressFree(reinterpret_cast<CUdeviceptr>(base_ptr_), mempool_.mempool_nbytes * VA_RESERVE_SCALE));
  }

}

void GenericAllocator::ExpandMemorySpace(
    const std::vector<PhyMem *> &phy_mem_list, size_t len) {
  if( (mapped_mem_list_.size() + len) * MEM_BLOCK_NBYTES > mempool_.mempool_nbytes * VA_RESERVE_SCALE) {
    LOG(INFO) << "[GenericAllocator] VA OMM";
    PrintOnCrash();
  }
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
                         len * MEM_BLOCK_NBYTES, &acc_desc, 1));
  cached_nbytes_ += len * MEM_BLOCK_NBYTES;
}
bool GenericAllocator::CheckState() {
  entry_list_.CheckState();
  free_list_small_.CheckState();
  free_list_large_.CheckState();
  return true;
}

}

