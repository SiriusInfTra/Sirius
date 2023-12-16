#include "mempool.h"
#include <glog/logging.h>
#include <initializer_list>
#include <iostream>
#include <tuple>

#ifdef NO_CUDA

#define CUDA_CALL(func)

#else

#include <cuda_runtime_api.h>
#define CUDA_CALL(func) do { \
  auto error = func; \
  if (error != cudaSuccess) { \
    LOG(FATAL) << #func << " " << cudaGetErrorString(error); \
    exit(EXIT_FAILURE); \
  } \
  } while (0)


#endif

namespace colserve::sta {


void MemPool::Free(MemPoolEntry *entry) {
  DLOG(INFO) << "[mempool] free " << entry->nbytes << ".";
  stat_->at(static_cast<size_t>(entry->mtype)).fetch_sub(entry->nbytes, std::memory_order_relaxed);
  bip::scoped_lock locker(*mutex_);
  DCHECK(CheckPoolWithoutLock());
  auto *next = GetNextEntry(segment_, entry, mem_entry_list_->end());
  auto *prev = GetPrevEntry(segment_, entry, mem_entry_list_->begin());
  auto next_free = next != nullptr && next->mtype == MemType::kFree;
  auto prev_free = prev != nullptr && prev->mtype == MemType::kFree;
  if (next_free && prev_free) {
    next->mtype = MemType::kMemTypeNum;
    freeblock_policy_->RemoveFreeBlock(next);

    size_t old_nbytes = prev->nbytes;
    prev->nbytes += entry->nbytes + next->nbytes;
    freeblock_policy_->NotifyUpdateFreeBlockNbytes(prev, old_nbytes);

    RemoveMemPoolEntry(entry);
    RemoveMemPoolEntry(next);
  } else if (next_free) {
    entry->nbytes += next->nbytes;
    entry->mtype = MemType::kFree;
    freeblock_policy_->AddFreeBlock(entry);

    next->mtype = MemType::kMemTypeNum;
    freeblock_policy_->RemoveFreeBlock(next);
    RemoveMemPoolEntry(next);
  } else if (prev_free) {
    size_t old_nbytes = prev->nbytes;
    prev->nbytes += entry->nbytes;
    freeblock_policy_->NotifyUpdateFreeBlockNbytes(prev, old_nbytes);

    RemoveMemPoolEntry(entry);
  } else {
    entry->mtype = MemType::kFree;
    freeblock_policy_->AddFreeBlock(entry);
  }
}
std::shared_ptr<PoolEntry>
MemPool::MakeSharedPtr(MemPoolEntry *entry) {
  auto *pool_entry = new PoolEntry{
      reinterpret_cast<std::byte *>(mem_pool_base_ptr_) + entry->addr_offset,
      entry->nbytes, entry->mtype};
  auto free = [this, entry](PoolEntry *pool_entry) {
    Free(entry);
    delete pool_entry;
  };
  return std::shared_ptr<PoolEntry>{pool_entry, free};
}
void MemPool::WaitSlaveExit() {
  if (master_) {
    auto getRefCount = [&] {
      bip::scoped_lock locker(*mutex_);
      return *ref_count_;
    };
    RefCount ref_count;
    while ((ref_count = getRefCount()) > 1) {
      LOG(INFO) << "[mempool] master wait slave shutdown, ref_count = "
                << ref_count << ".";
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
  }
}
bool MemPool::CheckPoolWithoutLock() {
  freeblock_policy_->CheckFreeList(mem_entry_list_->begin(),
                                   mem_entry_list_->end());
  DLOG(INFO) << "Check initial cond";
  CHECK_GE(mem_entry_list_->size(), 0);
  CHECK_EQ(mem_entry_list_->size(), mem_entry_table_->size());
  auto *first_entry = GetEntry(segment_, mem_entry_list_->front());
  CHECK_EQ(first_entry->addr_offset, 0)
      << "first entry addr_offset should be zero: " << *first_entry << ".";

  DLOG(INFO) << "Check block list";
  std::ptrdiff_t addr_offset = 0;
  for (auto iter = mem_entry_list_->cbegin(); iter != mem_entry_list_->cend();
       ++iter) {
    auto *entry = GetEntry(segment_, *iter);
    CHECK_EQ(entry->addr_offset, addr_offset)
        << "problem entry: " << *entry << " or prev: " << *GetPrevEntry(segment_, entry, mem_entry_list_->begin())
        << ".";
    CHECK(entry->mem_entry_pos == iter)
        << "mem_entry_pos mismatch: " << *entry << ".";
    auto iter_in_table = mem_entry_table_->find(entry->addr_offset);
    CHECK(iter_in_table != mem_entry_table_->cend())
        << "not found entry in entry_table: " << *entry << ".";
    auto *entry_in_table = GetEntry(segment_, iter_in_table->second);
    CHECK(entry == entry_in_table)
        << "entry: " << *entry << ", entry_in_table: " << *entry_in_table
        << ".";
    addr_offset += entry->nbytes;
  }
  CHECK_EQ(addr_offset, config_.cuda_memory_size);

  DLOG(INFO) << "Check block map";
  for (auto iter = mem_entry_table_->cbegin(); iter != mem_entry_table_->cend();
       ++iter) {
    auto *entry = GetEntry(segment_, iter->second);
    auto *entry_in_list = GetEntry(segment_, *entry->mem_entry_pos);
    CHECK_EQ(entry, entry_in_list)
        << "entry: " << *entry << ", entry_in_list:" << entry_in_list;
  }
  return true;
}
void MemPool::CopyFromToInternel(void *dst_dev_ptr, void *src_dev_ptr,
                                        size_t nbytes) {
  CUDA_CALL(cudaSetDevice(config_.cuda_device));
  CUDA_CALL(cudaMemcpyAsync(dst_dev_ptr, src_dev_ptr, nbytes, cudaMemcpyDefault,
                            cuda_memcpy_stream_));
  CUDA_CALL(cudaStreamSynchronize(cuda_memcpy_stream_));
}
MemPool::MemPool(MemPoolConfig config, bool cleanup, bool observe, FreeListPolicyType policy_type)
    : config_(std::move(config)), mem_pool_base_ptr_(nullptr), observe_(observe) {
  if (std::getenv("USER") != nullptr) {
    config_.shared_memory_name =
        config_.shared_memory_name + "_" + std::getenv("USER");
  } else {
    config_.shared_memory_name =
        config_.shared_memory_name + "_" + std::to_string(getuid());
  }
  if (cleanup) {
    bip::shared_memory_object::remove(config_.shared_memory_name.c_str());
  }
  segment_ = bip::managed_shared_memory{bip::open_or_create,
                                        config_.shared_memory_name.c_str(),
                                        config_.shared_memory_size};
  auto atomic_init = [&] {
    mutex_ =
        segment_.find_or_construct<bip::interprocess_mutex>("ShareMutex")();
    mem_entry_table_ = segment_.find_or_construct<EntryAddrTable>(
        "MemEntryTable")(segment_.get_segment_manager());
    mem_entry_list_ = segment_.find_or_construct<EntryList>("MemEntryList")(
        segment_.get_segment_manager());
    ref_count_ = segment_.find_or_construct<RefCount>("RefCount")(0);
    cuda_mem_handle_ =
        segment_.find_or_construct<CUDA_TYPE(cudaIpcMemHandle_t)>(
            "CudaMemHandle")();
    switch (policy_type) {

      case FreeListPolicyType::kNextFit:
        freeblock_policy_ = new NextFitPolicy(segment_);
        LOG(INFO) << "Init next-fit policy";
        break;
      case FreeListPolicyType::kFirstFit:
        freeblock_policy_ = new FirstFitPolicy(segment_);
        LOG(INFO) << "Init first-fit policy";
        break;
      case FreeListPolicyType::kBestFit:
        freeblock_policy_ = new BestFitPolicy(segment_);
        LOG(INFO) << "Init best-fit policy";
        break;
      default:
        LOG(FATAL) << "unknown policy_type: " << static_cast<int>(policy_type);
    }
    freeblock_policy_ = new FirstFitPolicy(segment_);
    stat_ = segment_.find_or_construct<StatMap>("StatMap")();
  };
  segment_.atomic_func(atomic_init);
  if (observe_)  { return; }
  CUDA_CALL(cudaSetDevice(config.cuda_device));
  CUDA_CALL(cudaStreamCreate(&cuda_memcpy_stream_));
  bip::scoped_lock locker(*mutex_);
  master_ = (*ref_count_)++ == 0;
  if (master_) {
    auto *entry = CreateMemPoolEntry(0, config_.cuda_memory_size,
                                     MemType::kFree, mem_entry_list_->end());
    freeblock_policy_->InitMaster(entry);
    CUDA_CALL(cudaSetDevice(config_.cuda_device));
    CUDA_CALL(cudaMalloc(reinterpret_cast<void **>(&mem_pool_base_ptr_),
                         config_.cuda_memory_size));
    CUDA_CALL(cudaIpcGetMemHandle(cuda_mem_handle_, mem_pool_base_ptr_));
    LOG(INFO) << "[mempool] init master, base_ptr = " << std::hex
              << mem_pool_base_ptr_ << ", shm = " << config_.shared_memory_name
              << ".";
  } else {
    freeblock_policy_->InitSlave();
    CUDA_CALL(cudaIpcOpenMemHandle(
        reinterpret_cast<void **>(&mem_pool_base_ptr_), *cuda_mem_handle_,
        cudaIpcMemLazyEnablePeerAccess));
    LOG(INFO) << "[mempool] init slave, base_ptr = " << std::hex
              << mem_pool_base_ptr_ << ", shm = " << config_.shared_memory_name
              << ".";
  }
}

MemPool::~MemPool() {
  if (observe_) { return; }
  if (master_) {
    WaitSlaveExit();
    bip::shared_memory_object::remove(config_.shared_memory_name.c_str());
    LOG(INFO) << "[mempool] free master.";
  } else {
    bip::scoped_lock locker(*mutex_);
    --(*ref_count_);
    LOG(INFO) << "[mempool] free slave.";
  }
}
void MemPool::CheckPool() {
  bip::scoped_lock locker(*mutex_);
  CheckPoolWithoutLock();
}

void MemPool::RemoveMemPoolEntry(MemPoolEntry *entry) {
  mem_entry_list_->erase(entry->mem_entry_pos);
  mem_entry_table_->erase(entry->addr_offset);
  segment_.deallocate(entry);
}
MemPoolEntry *
MemPool::CreateMemPoolEntry(std::ptrdiff_t addr_offset,
                                           std::size_t nbytes, MemType mtype,
                                           EntryListIterator insert_pos) {
  auto *entry =
      reinterpret_cast<MemPoolEntry *>(segment_.allocate(sizeof(MemPoolEntry)));
  entry->nbytes = nbytes;
  entry->addr_offset = addr_offset;
  entry->mtype = mtype;
  entry->mem_entry_pos =
      mem_entry_list_->insert(insert_pos, GetHandle(segment_, entry));
  mem_entry_table_->insert(
      std::make_pair(entry->addr_offset, GetHandle(segment_, entry)));
  return entry;
}
std::shared_ptr<PoolEntry> MemPool::Alloc(std::size_t nbytes,
                                                         MemType mtype) {
  DLOG(INFO) << "[mempool] alloc " << nbytes << ".";
  if (nbytes == 0) { 
    return std::shared_ptr<PoolEntry>(new PoolEntry{nullptr, 0, mtype});
  }
  nbytes = (nbytes + 1023) / 1024 * 1024;
  stat_->at(static_cast<size_t>(mtype)).fetch_add(nbytes, std::memory_order_relaxed);
  bip::scoped_lock locker(*mutex_);  
  DCHECK(CheckPoolWithoutLock());
  auto *entry = freeblock_policy_->GetFreeBlock(nbytes);
  if (entry == nullptr) {
    DumpSummaryWithoutLock();
    LOG(FATAL) << "[mempool] fail to alloc " << detail::ByteDisplay(nbytes) << ".";
  }
  CHECK(entry->mtype == MemType::kFree);
  entry->mtype = mtype;
  freeblock_policy_->RemoveFreeBlock(entry);

  if (entry->nbytes > nbytes) {
    auto insert_pos = entry->mem_entry_pos;
    ++insert_pos;
    auto *free_entry =
        CreateMemPoolEntry(entry->addr_offset + nbytes, entry->nbytes - nbytes,
                           MemType::kFree, insert_pos);
    freeblock_policy_->AddFreeBlock(free_entry);
    entry->nbytes = nbytes;
  }
  return MakeSharedPtr(entry);
}

void MemPool::CopyFromTo(std::shared_ptr<PoolEntry> src,
                         std::shared_ptr<PoolEntry> dst) {
  if (src == nullptr || dst == nullptr)
    return;
  if (src->addr == dst->addr)
    return;
  CopyFromToInternel(dst->addr, src->addr, std::min(src->nbytes, dst->nbytes));
}

std::unordered_map<MemType, std::unordered_map<UsageStat, size_t>>
MemPool::GetUsageWithoutLock() {
  std::unordered_map<MemType, std::unordered_map<UsageStat, size_t>> usages;
  for (auto handle : *mem_entry_list_) {
    auto* entry = GetEntry(segment_, handle);
    auto& usages_mtype = usages[entry->mtype];
    {
      auto& maxNbytes = usages_mtype[UsageStat::kMaxNBytes];
      maxNbytes = std::max(maxNbytes, entry->nbytes);
    }
    {
      auto& minNbytes = usages_mtype[UsageStat::kMinNBytes];
      minNbytes = std::max(minNbytes, entry->nbytes);
    }
    {
      auto& count = usages_mtype[UsageStat::kCount];
      ++count;
    }
    {
      auto& totalNBytes = usages_mtype[UsageStat::kTotalNBytes];
      totalNBytes += entry->nbytes;
    }
  }
  return usages;
}

void MemPool::DumpSummaryWithoutLock() {
  std::initializer_list<std::tuple<std::string, MemType>> mtype_list = {
      {"free", MemType::kFree},
      {"infer", MemType::kInfer},
      {"train", MemType::kTrain}};
  auto usages = GetUsageWithoutLock();
  std::cout << "now dump memory pool summary" << std::endl;
  LOG(INFO) << "---------- mempool summary ----------";
  for (auto&& [name, mtype] : mtype_list) {
    auto& usages_mtype = usages[mtype];
    LOG(INFO) << name << " max: "
              << detail::ByteDisplay(usages_mtype[UsageStat::kMaxNBytes]);
    LOG(INFO) << name << " max: "
              << detail::ByteDisplay(usages_mtype[UsageStat::kMinNBytes]);
    LOG(INFO) << name << " sum: "
              << detail::ByteDisplay(usages_mtype[UsageStat::kTotalNBytes]);
    LOG(INFO) << name << " cnt: "
              << usages_mtype[UsageStat::kCount];
    if (usages_mtype[UsageStat::kCount] == 0) { continue; }
    LOG(INFO) << name << " avg: "
              << detail::ByteDisplay(usages_mtype[UsageStat::kTotalNBytes] /
                                     usages_mtype[UsageStat::kCount]);
  }
  google::FlushLogFiles(google::INFO);
}



MemPoolEntry* BestFitPolicy::GetFreeBlock(size_t nbytes) {
  auto iter = free_entry_table->lower_bound(nbytes);
  if (iter == free_entry_table->cend()) {
    return nullptr;
  }
  return GetEntry(segment_, iter->second);
}

void BestFitPolicy::NotifyUpdateFreeBlockNbytes(MemPoolEntry* entry,
                                                size_t old_nbytes) {
  CHECK(entry->mtype == MemType::kFree)
      << "update not free entry: " << *entry << " in free table";
  free_entry_table->erase(entry->freetable_pos);
  entry->freetable_pos = free_entry_table->insert(
      std::make_pair(entry->nbytes, GetHandle(segment_, entry)));
}

void BestFitPolicy::RemoveFreeBlock(MemPoolEntry* entry) {
  CHECK(entry->mtype != MemType::kFree)
      << "remove free entry: " << *entry << " from free table";
  free_entry_table->erase(entry->freetable_pos);
}

void BestFitPolicy::AddFreeBlock(MemPoolEntry* entry) {
  CHECK(entry->mtype == MemType::kFree)
      << "add not free entry: " << *entry << " to free table";
  free_entry_table->insert(
      std::make_pair(entry->nbytes, GetHandle(segment_, entry)));
}

void BestFitPolicy::CheckFreeList(const EntryListIterator& begin,
                                  const EntryListIterator& end) {
  DLOG(INFO) << "Check freelist";
  std::unordered_set<std::ptrdiff_t> free_set;
  for (auto iter = free_entry_table->cbegin(); iter != free_entry_table->cend();
       iter++) {
    auto* entry = GetEntry(segment_, iter->second);
    free_set.insert(entry->addr_offset);
    CHECK_EQ(entry->nbytes, iter->first)
        << "entry nbytes not match: " << *entry << ".";
    CHECK(entry->mtype == MemType::kFree)
        << "entry in freetable but not free: " << *entry << ".";
  }

  for (auto iter = begin; iter != end; iter++) {
    auto* entry = GetEntry(segment_, *iter);
    if (entry->mtype == MemType::kFree &&
        free_set.find(entry->addr_offset) == free_set.cend()) {
      CHECK(entry->mtype == MemType::kFree)
          << "entry in free but not in free list: " << *entry << ".";
    }
  }
}

void BestFitPolicy::InitMaster(MemPoolEntry* free_entry) {
  AddFreeBlock(free_entry);
}

BestFitPolicy::BestFitPolicy(shared_memory& segment)
    : FreeListPolicy(segment) {
  free_entry_table = segment.find_or_construct<EntrySizeTable>("FreeTable")(
      segment_.get_segment_manager());
}

void NextFitPolicy::NotifyUpdateFreeBlockNbytes(MemPoolEntry* entry,
                                                size_t old_nbytes) {
  CHECK(entry->mtype == MemType::kFree)
      << "try update not free entry in freelist: " << *entry << ".";
}

void NextFitPolicy::AddFreeBlock(MemPoolEntry* entry) {
  CHECK(entry->mtype == MemType::kFree)
      << "try add not free entry to freelist: " << *entry << ".";
  entry->freelist_pos =
      freelist_->insert(freelist_->end(), GetHandle(segment_, entry));
}

void NextFitPolicy::RemoveFreeBlock(MemPoolEntry* entry) {
  CHECK(entry->mtype != MemType::kFree)
      << "try to remove free entry from freelist: " << *entry << ".";
  if (entry->freelist_pos == *freelist_pos_) {
    *freelist_pos_ = freelist_->erase(entry->freelist_pos);
  } else {
    freelist_->erase(entry->freelist_pos);
  }
}

void NextFitPolicy::CheckFreeList(const EntryListIterator& begin,
                                  const EntryListIterator& end) {
  DLOG(INFO) << "Check freelist";
  std::unordered_set<std::ptrdiff_t> free_set;
  for (auto iter = freelist_->cbegin(); iter != freelist_->cend(); iter++) {
    auto* entry = GetEntry(segment_, *iter);
    free_set.insert(entry->addr_offset);
    CHECK(entry->mtype == MemType::kFree)
        << "entry in freelist but not free: " << *entry << ".";
  }

  for (auto iter = begin; iter != end; iter++) {
    auto* entry = GetEntry(segment_, *iter);
    if (entry->mtype == MemType::kFree &&
        free_set.find(entry->addr_offset) == free_set.cend()) {
      CHECK(entry->mtype == MemType::kFree)
          << "entry in free but not in free list: " << *entry << ".";
    }
  }
}

void NextFitPolicy::DumpFreeList(std::ostream& stream,
                                 const EntryListIterator& begin,
                                 const EntryListIterator& end) {
  stream << "start,len,allocated,next,prev,mtype" << std::endl;
  for (const auto& element : *freelist_) {
    auto* entry = GetEntry(segment_, element);
    auto* prev = GetPrevEntry(segment_, entry, begin);
    auto* next = GetNextEntry(segment_, entry, end);
    stream << entry->addr_offset << "," << entry->nbytes << ","
           << static_cast<int>(entry->mtype) << ","
           << (prev ? next->addr_offset : -1) << ","
           << (prev ? prev->addr_offset : -1) << ","
           << static_cast<unsigned>(entry->mtype) << std::endl;
  }
}

MemPoolEntry* FirstFitPolicy::GetFreeBlock(size_t nbytes) {
  for (auto handle : *freelist_) {
    auto* entry = GetEntry(segment_, handle);
    CHECK(entry->mtype == MemType::kFree)
        << "not free entry in freelist: " << *entry << ".";
    if (entry->nbytes >= nbytes) {
      return entry;
    }
  }
  return nullptr;
}

FreeListPolicyType getFreeListPolicy(const std::string& s) {
  if (s == "first-fit") {
    return colserve::sta::FreeListPolicyType::kFirstFit;
  } else if (s == "next-fit") {
    return colserve::sta::FreeListPolicyType::kNextFit;
  } else if (s == "best-fit") {
    return colserve::sta::FreeListPolicyType::kBestFit;
  } else {
    LOG(FATAL) << "unknown free list policy: " << s;
  }
}
}  // namespace colserve::sta
