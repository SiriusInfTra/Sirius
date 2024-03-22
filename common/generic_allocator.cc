#include <common/generic_allocator.h>
#include <common/mempool.h>

#include <glog/logging.h>
#include <cstddef>
#include <cstring>
#include <string>


namespace colserve::sta {
void EntryList::LinkNewEntry(MemEntry *entry) {
  if (entry_list_->empty()) {
    CHECK_EQ(entry->addr_offset, 0);
  } else {
    CHECK_EQ(entry->addr_offset, entry_list_->back().ptr()->addr_offset + entry_list_->back().ptr()->nbytes);
  }

  entry->pos_entrytable =
      entry_by_addr->insert(std::make_pair(entry->addr_offset, shm_handle<MemEntry>(entry))).first;
  entry->pos_entrylist = entry_list_->insert(entry_list_->cend(), shm_handle<MemEntry>(entry));
}
MemEntry *
EntryList::GetEntry(std::ptrdiff_t addr_offset) {
  auto iter = entry_by_addr->find(addr_offset);
  if (iter == entry_by_addr->cend()) {
    return nullptr;
  }
  auto *entry = iter->second.ptr();
  CHECK(iter == entry->pos_entrytable); 
  return entry;
}

MemEntry *
EntryList::GetPrevEntry(MemEntry *entry) {
  auto iter = entry->pos_entrylist;
  if (iter == entry_list_->cbegin()) {
    return nullptr;
  }
  return std::prev(iter)->ptr();
}
MemEntry *
EntryList::GetNextEntry(MemEntry *entry) {
  auto iter = std::next(entry->pos_entrylist);
  if (iter == entry_list_->cend()) {
    return nullptr;
  }
  return iter->ptr();
}

MemEntry *
EntryList::SplitEntry(MemEntry *origin_entry,
                                         size_t remain) {
  CHECK_GT(origin_entry->nbytes, remain);
  bool insert_success;
  auto *entry_split = reinterpret_cast<MemEntry *>(MemPool::Get().GetSharedMemory().allocate(sizeof(MemEntry)));
  entry_split->is_free  = origin_entry->is_free;
  entry_split->is_small = origin_entry->is_small;
  entry_split->is_train = origin_entry->is_train;
  
  /* [origin: remain] [split: nbytes - remain] */
  entry_split->nbytes = origin_entry->nbytes - remain;
  entry_split->addr_offset = origin_entry->addr_offset + remain;
  entry_split->pos_entrylist =
      entry_list_->insert(std::next(origin_entry->pos_entrylist), entry_split);
  entry_split->is_small = origin_entry->is_small;
  origin_entry->nbytes = remain;
  std::tie(entry_split->pos_entrytable, insert_success) = entry_by_addr->insert(
      std::make_pair(entry_split->addr_offset, shm_handle<MemEntry>(entry_split)));
  CHECK(insert_success);
  CHECK(!alloc_conf::STRICT_CHECK_STATE || CheckState());
  return entry_split;
}

MemEntry *
EntryList::MergeMemEntry(MemEntry *first_entry,
                                         MemEntry *secound_entry) {
  CHECK_EQ(first_entry->addr_offset + first_entry->nbytes,
           secound_entry->addr_offset);
  CHECK_EQ(GetNextEntry(first_entry), secound_entry);
  CHECK_EQ(first_entry->is_free, secound_entry->is_free);
  CHECK_EQ(first_entry->is_train, secound_entry->is_train);
  CHECK_EQ(first_entry->is_small, secound_entry->is_small);
  first_entry->nbytes += secound_entry->nbytes;
  entry_list_->erase(secound_entry->pos_entrylist);
  entry_by_addr->erase(secound_entry->pos_entrytable);
  memset(secound_entry, 63, sizeof(MemEntry));
  MemPool::Get().GetSharedMemory().deallocate(secound_entry);
  CHECK(!alloc_conf::STRICT_CHECK_STATE || CheckState());
  return first_entry;
}
void EntryList::DumpMemEntryList(std::ostream &out) {
  out << "start,len,next,prev,is_free,is_train,is_small"
      << "\n";
  for (auto handle : *entry_list_) {
    auto *entry = handle.ptr();
    auto *prev = GetPrevEntry(entry);
    auto *next = GetNextEntry(entry);
    out << entry->addr_offset << "," << entry->nbytes << ","
        << (next ? next->addr_offset : -1) << ","
        << (prev ? prev->addr_offset : -1) << ","
        << entry->is_free << ","
        << entry->is_train << ","
        << entry->is_small << "\n";
  }
  out << std::flush;
}
bool EntryList::CheckState() {
  if constexpr (alloc_conf::DUMP_BEFORE_CHECK) {
    LOG(INFO) << log_prefix_ << "Dump entry_list.";
    DumpMemEntryList(std::cerr);
  }
  for (auto handle : *entry_list_) {
    auto *entry = handle.ptr();
    if (auto *prev = GetPrevEntry(entry); prev) {
      CHECK_EQ(prev->addr_offset + prev->nbytes, entry->addr_offset);
    }
    if (auto *next = GetNextEntry(entry); next) {
      CHECK_EQ(entry->addr_offset + entry->nbytes, next->addr_offset);
    }
  }

  return true;
}


FreeList::FreeList(EntryList &list_index, bool is_small, const std::string &log_prefix, Belong policy)
    : list_index_(list_index), is_small_(is_small), policy_(policy), log_prefix_(log_prefix) {
          auto &shared_memory = MemPool::Get().GetSharedMemory();
  auto atomic_init = [&] {
    std::string name = "FL_entry_by_nbytes_" + std::to_string(is_small) + "_" + std::to_string(static_cast<size_t>(policy));
    entry_by_nbytes_ = shared_memory.find_or_construct<entry_nbytes_map>(name.c_str())(shared_memory.get_segment_manager());
  };
  shared_memory.atomic_func(atomic_init);
}



MemEntry *FreeList::PopFreeEntry(size_t nbytes, bool do_split) {
  auto iter = entry_by_nbytes_->lower_bound(nbytes);
  if (iter == entry_by_nbytes_->cend()) {
    return nullptr;
  }
  auto *free_entry = iter->second.ptr();
  CHECK_GE(free_entry->nbytes, nbytes);
  if (do_split && free_entry->nbytes > nbytes ) {
    auto split_entry = list_index_.SplitEntry(free_entry, nbytes);
    split_entry->pos_freelist = entry_by_nbytes_->insert(
        std::make_pair(split_entry->nbytes, shm_handle<MemEntry>(split_entry)));
  }
  entry_by_nbytes_->erase(free_entry->pos_freelist);
  free_entry->is_free = false;
  CHECK_EQ(free_entry->is_small, is_small_);
  CHECK(!alloc_conf::STRICT_CHECK_STATE || CheckState());
  return free_entry;
}


MemEntry *FreeList::PopFreeEntry(MemEntry *free_entry) {
  CHECK_EQ(free_entry->is_small, is_small_);
  CHECK(free_entry->is_free);
  entry_by_nbytes_->erase(free_entry->pos_freelist);
  free_entry->is_free = false;
  CHECK_EQ(free_entry->is_small, is_small_);
  CHECK(!alloc_conf::STRICT_CHECK_STATE || CheckState());
  if (!is_small_) {
    CHECK_GE(free_entry->nbytes, SMALL_BLOCK_NBYTES);
  }
  return free_entry;
}


MemEntry *FreeList::PopFreeEntryLarge(size_t nbytes) {
    if (entry_by_nbytes_->empty()) { return nullptr; }
    auto largest_entry = std::prev(entry_by_nbytes_->cend())->second.ptr();
    if (largest_entry->nbytes < nbytes) { return nullptr; }
    entry_by_nbytes_->erase(largest_entry->pos_freelist);
    largest_entry->is_free = false;
    CHECK_EQ(largest_entry->is_small, is_small_);
    CHECK(!alloc_conf::STRICT_CHECK_STATE || CheckState());
    return largest_entry;
  }


MemEntry* FreeList::PushFreeEntry(MemEntry *entry) {
  CHECK_EQ(entry->is_free, false);
  CHECK_EQ(entry->is_small, is_small_);
  entry->is_free = true;
  if (auto prev_entry = list_index_.GetPrevEntry(entry); prev_entry 
    && prev_entry->is_free 
    && prev_entry->is_small == entry->is_small
    && (policy_ == Belong::kTrain || (policy_ == Belong::kInfer && !entry->is_train && !prev_entry->is_train))
  ) {
    entry_by_nbytes_->erase(prev_entry->pos_freelist);
    entry = list_index_.MergeMemEntry(prev_entry, entry);
  }
  if (auto next_entry = list_index_.GetNextEntry(entry); next_entry 
    && next_entry->is_free 
    && next_entry->is_small == entry->is_small
    && (policy_ == Belong::kTrain || (policy_ == Belong::kInfer && !entry->is_train && !next_entry->is_train))
  ) {
    entry_by_nbytes_->erase(next_entry->pos_freelist);
    entry = list_index_.MergeMemEntry(entry, next_entry);
  }

  entry->pos_freelist =
      entry_by_nbytes_->insert(std::make_pair(entry->nbytes, shm_handle<MemEntry>(entry)));
  // if (!is_small_) {
  //   CHECK_GE(entry->nbytes, SMALL_BLOCK_NBYTES);
  // }
  CHECK(!alloc_conf::STRICT_CHECK_STATE || CheckState());
  return entry;
}


void FreeList::DumpFreeList(std::ostream &out) {
  out << "start,len,next,prev,is_free,is_small"
      << "\n";
  for (auto &&[nbytes, shm_handle] : *entry_by_nbytes_) {
    auto *entry = shm_handle.ptr();
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
  if constexpr (alloc_conf::DUMP_BEFORE_CHECK) {
    LOG(INFO) << log_prefix_ << "Dump free_list "<< (is_small_ ? "small" : "large") <<".";
    DumpFreeList(std::cerr);
  }
  for (auto &&[nbytes, shm_handle] : *entry_by_nbytes_) {
    auto *entry = shm_handle.ptr();
    CHECK_EQ(entry->is_free, true);
    CHECK_EQ(entry->nbytes, nbytes);
    CHECK_EQ(entry->is_small, is_small_);
  }
  return true;
}


GenericAllocator::GenericAllocator(MemPool &mempool, Belong policy, bip::scoped_lock<bip::interprocess_mutex> &lock)
    : mempool_(mempool), 
      log_prefix_("[" + ToString(policy) + "] "), 
      entry_list_(log_prefix_, policy), 
      free_list_small_(entry_list_, true, log_prefix_, policy),
      free_list_large_(entry_list_, false, log_prefix_, policy), 
      policy_(policy) {
  CHECK(lock.owns());
  CU_CALL(cuMemAddressReserve(reinterpret_cast<CUdeviceptr *>(&base_ptr_),
                              mempool_.mempool_nbytes * VA_RESERVE_SCALE, MEM_BLOCK_NBYTES, 0, 0));
  LOG(INFO) << log_prefix_ << "dev_ptr = " << base_ptr_ << ".";
}


GenericAllocator::~GenericAllocator() {
  if (!mapped_mem_list_.empty()) {
    if (mempool_.IsInit()) { DumpState(); }
    int ignore;
    if (cuDriverGetVersion(&ignore) != CUDA_ERROR_DEINITIALIZED) {
      CU_CALL(cuMemAddressFree(reinterpret_cast<CUdeviceptr>(base_ptr_), mempool_.mempool_nbytes * VA_RESERVE_SCALE));
    }
  }

}

bool GenericAllocator::CheckState() {
  entry_list_.CheckState();
  free_list_small_.CheckState();
  free_list_large_.CheckState();
  return true;
}

}

