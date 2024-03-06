#include "mempool.h"
#include <glog/logging.h>
#include <initializer_list>
#include <ios>
#include <iostream>
#include <tuple>
#include <fstream>

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

std::ostream &operator<<(std::ostream &os, const MemType &mtype) {
  switch (mtype) {
    case MemType::kFree:
      os << "MemType::kFree";
      return os;
    case MemType::kTrainLocalFree:
      os << "MemType::kTrainLocalFree";
      return os;
    case MemType::kInfer:
      os << "MemType::kInfer";
      return os;
    case MemType::kTrain:
      os << "MemType::kTrain";
      return os;
    case MemType::kTrainAll:
      os << "MemType::kTrainAll";
      return os;
    default:
      LOG(FATAL) << "unkown MemType " << static_cast<size_t>(mtype);
      return os;
  }
}

void MemPool::Free(MemPoolEntry *entry) {
  bip::scoped_lock locker(*mutex_);
  DCHECK(CheckPoolWithoutLock());
  FreeWithoutLock(entry);
}

MemPoolEntry* MemPool::FreeWithoutLock(MemPoolEntry *entry) {
  DLOG(INFO) << "[mempool] free " << entry->nbytes << ".";
  stat_->at(static_cast<size_t>(entry->mtype)).fetch_sub(entry->nbytes, std::memory_order_relaxed);
  if (entry->mtype == MemType::kTrainLocalFree) {
    stat_->at(static_cast<size_t>(MemType::kTrainAll)).fetch_sub(entry->nbytes, std::memory_order_relaxed);
  }
  auto *next = GetNextEntry(segment_, entry, mem_entry_list_->end());
  auto *prev = GetPrevEntry(segment_, entry, mem_entry_list_->begin());
  auto next_free = next != nullptr && next->IsMergableFree(entry->mtype);
  auto prev_free = prev != nullptr && prev->IsMergableFree(entry->mtype);
  if (next_free && prev_free) {
    freeblock_policy_->RemoveFreeBlock(next);

    size_t old_nbytes = prev->nbytes;
    prev->nbytes += entry->nbytes + next->nbytes;
    freeblock_policy_->NotifyUpdateFreeBlockNbytes(prev, old_nbytes);
    stat_->at(static_cast<size_t>(prev->mtype)).fetch_add(entry->nbytes, std::memory_order_relaxed);

    RemoveMemPoolEntry(entry);
    RemoveMemPoolEntry(next);
    return prev;
  } else if (next_free) {
    auto old_entry_nbytes = entry->nbytes;
    entry->nbytes += next->nbytes;
    entry->SetAsFree();
    freeblock_policy_->AddFreeBlock(entry);
    stat_->at(static_cast<size_t>(entry->mtype)).fetch_add(old_entry_nbytes, std::memory_order_relaxed);

    freeblock_policy_->RemoveFreeBlock(next);
    RemoveMemPoolEntry(next);
    return entry;
  } else if (prev_free) {
    size_t old_nbytes = prev->nbytes;
    prev->nbytes += entry->nbytes;
    freeblock_policy_->NotifyUpdateFreeBlockNbytes(prev, old_nbytes);
    stat_->at(static_cast<size_t>(prev->mtype)).fetch_add(entry->nbytes, std::memory_order_relaxed);

    RemoveMemPoolEntry(entry);
    return prev;
  } else {
    entry->SetAsFree();
    freeblock_policy_->AddFreeBlock(entry);
    stat_->at(static_cast<size_t>(entry->mtype)).fetch_add(entry->nbytes, std::memory_order_relaxed);
    return entry;
  }
}

void MemPool::FreeLocals(MemType mtype) {
  CHECK(mtype == MemType::kTrainLocalFree);
  bip::scoped_lock locker(*mutex_);
  
  DCHECK(CheckPoolWithoutLock());
  freeblock_policy_->RemoveLocalFreeBlocks(mtype, [this](MemPoolEntry* entry) {
    this->FreeWithoutLock(entry);
  });
}

std::shared_ptr<PoolEntry>
MemPool::MakeSharedPtr(MemPoolEntry *entry) {
  auto *pool_entry = new PoolEntry{
      reinterpret_cast<std::byte *>(mem_pool_base_ptr_) + entry->addr_offset,
      entry->nbytes, entry->mtype};
  CHECK_NE(mem_pool_base_ptr_, nullptr);
  CHECK_GE(static_cast<int64_t>(entry->addr_offset), 0);
  CHECK_LE(static_cast<size_t>(entry->addr_offset) + entry->nbytes, this->config_.cuda_memory_size);
  CHECK_EQ(reinterpret_cast<size_t>(pool_entry->addr) & (detail::alignment - 1), 0) 
    << "unaligned entry->addr " << pool_entry->addr;

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
  auto *gpu_id = std::getenv("CUDA_VISIBLE_DEVICES");
  CHECK(gpu_id != nullptr);
  if (std::getenv("USER") != nullptr) {
    config_.shared_memory_name =
        config_.shared_memory_name + "_" + std::getenv("USER") + "-" + gpu_id;
  } else {
    config_.shared_memory_name =
        config_.shared_memory_name + "_" + std::to_string(getuid()) + "-" + gpu_id;
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
    stat_->at(static_cast<size_t>(MemType::kFree)).fetch_add(config_.cuda_memory_size, std::memory_order_relaxed);
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

#if 0
std::shared_ptr<PoolEntry> MemPool::Alloc(std::size_t nbytes,
                                          MemType mtype) {
  DLOG(INFO) << "[mempool] alloc " << detail::ByteDisplay(nbytes) << ".";
  if (nbytes == 0) { 
    return std::shared_ptr<PoolEntry>(new PoolEntry{nullptr, 0, mtype});
  }
  // nbytes = (nbytes + 1023) / 1024 * 1024;
  auto t0 = std::chrono::steady_clock::now();
  nbytes = detail::GetAlignedNbytes(nbytes);
  bip::scoped_lock locker(*mutex_);
  DCHECK(CheckPoolWithoutLock());
  auto *entry = freeblock_policy_->GetFreeBlock(nbytes, mtype, true, true);
  if (entry == nullptr) {
    DumpSummaryWithoutLock();
    // DumpBlockListWithoutLock(); // TODO: buggy?
    LOG(FATAL) << "[mempool] fail to alloc " << mtype << " " << detail::ByteDisplay(nbytes) << ".";
  }
  CHECK(entry->IsAvailableFree(mtype));
  freeblock_policy_->RemoveFreeBlock(entry);
  stat_->at(static_cast<size_t>(mtype)).fetch_add(nbytes, std::memory_order_relaxed);
  stat_->at(static_cast<size_t>(entry->mtype)).fetch_sub(nbytes, std::memory_order_relaxed);

  auto free_mtype = entry->mtype;
  entry->mtype = mtype;

  if (mtype == MemType::kTrain && free_mtype == MemType::kFree) {
    stat_->at(static_cast<size_t>(MemType::kTrainAll)).fetch_add(nbytes, std::memory_order_relaxed);
  }

  if (entry->nbytes > nbytes) {
    auto insert_pos = entry->mem_entry_pos;
    ++insert_pos;
    auto *free_entry =
        CreateMemPoolEntry(entry->addr_offset + nbytes, entry->nbytes - nbytes,
                           free_mtype, insert_pos);
    freeblock_policy_->AddFreeBlock(free_entry);
    entry->nbytes = nbytes;
  }
  CHECK(entry->mtype == MemType::kInfer || entry->mtype == MemType::kTrain);
  auto t1 = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
  // if (duration.count() > 50) {
  //   LOG(WARNING) << "[mempool] alloc " << mtype << " " << detail::ByteDisplay(nbytes) << " " << duration.count() << " us.";
  // }
  return MakeSharedPtr(entry);
}

// TODO: check if this is buggy
#else
std::shared_ptr<PoolEntry> MemPool::Alloc(std::size_t nbytes,
                                          MemType mtype) {
  DLOG(INFO) << "[mempool] alloc " << detail::ByteDisplay(nbytes) << ".";
  if (nbytes == 0) { 
    return std::shared_ptr<PoolEntry>(new PoolEntry{nullptr, 0, mtype});
  }
  // auto t0 = std::chrono::steady_clock::now();
  nbytes = detail::GetAlignedNbytes(nbytes);
  bip::scoped_lock locker(*mutex_);
  DCHECK(CheckPoolWithoutLock());
  MemPoolEntry *entry = nullptr;
  bool train_over_threshold = false;
  bool alloc_by_merge = false;
  size_t alloc_nbytes = nbytes;
  if (mtype == MemType::kTrain) {
    entry = freeblock_policy_->GetFreeBlock(alloc_nbytes, mtype, true, false);
    if (entry == nullptr) { // try to find in global
      alloc_nbytes = detail::GetAlignedNbytes(nbytes, detail::train_alloc_threshold);
      entry = freeblock_policy_->GetFreeBlock(alloc_nbytes, mtype, false, true);
      if (entry == nullptr) { // fail, fallback to smaller alignment
        alloc_nbytes = detail::GetAlignedNbytes(nbytes, detail::train_alloc_threshold_small);
        entry = freeblock_policy_->GetFreeBlock(alloc_nbytes, mtype, false, true);
      }
      if (entry == nullptr) { // merge local with global
        auto merge_entry = freeblock_policy_->GetFreeBlockByMerge(alloc_nbytes, mtype, mem_entry_list_);
        if (merge_entry != nullptr) {
          alloc_by_merge = true;
          freeblock_policy_->RemoveFreeBlock(merge_entry);
          // LOG(INFO) << "[mempool] merge local with global " << detail::ByteDisplay(alloc_nbytes) << ".";
          entry = FreeWithoutLock(merge_entry);
          CHECK(entry->mtype == MemType::kFree && entry->nbytes >= alloc_nbytes)
            << "entry: " << entry->mtype << " nbytes " << entry->nbytes << " alloc_nbytes " << alloc_nbytes;
          alloc_nbytes = entry->nbytes;
        }
      }
      train_over_threshold = alloc_nbytes > nbytes;
    }
  } else { // infer
    entry = freeblock_policy_->GetFreeBlock(alloc_nbytes, mtype, true, true);
  }
  if (entry == nullptr) {
    DumpSummaryWithoutLock();
    DumpBlockListWithoutLock(); // TODO: buggy?
    LOG(FATAL) << "[mempool] fail to alloc " << mtype << " " << detail::ByteDisplay(alloc_nbytes) << ".";
  }

  CHECK(entry->IsAvailableFree(mtype)) << " " << entry << " mtype " << mtype << " entry->mtype " << entry->mtype;
  // if (train_over_threshold && entry) {
  //   LOG(INFO) << "[mempool] alloc " << mtype << " " << detail::ByteDisplay(alloc_nbytes) << " "
  //             << entry << " " << entry->mtype;
  // }
  freeblock_policy_->RemoveFreeBlock(entry);
  stat_->at(static_cast<size_t>(mtype)).fetch_add(alloc_nbytes, std::memory_order_relaxed);
  stat_->at(static_cast<size_t>(entry->mtype)).fetch_sub(alloc_nbytes, std::memory_order_relaxed);

  auto free_mtype = entry->mtype;
  entry->mtype = mtype;

  if (mtype == MemType::kTrain && free_mtype == MemType::kFree) {
    stat_->at(static_cast<size_t>(MemType::kTrainAll)).fetch_add(alloc_nbytes, std::memory_order_relaxed);
  }

  auto split_free_entry = [this, mtype] (MemPoolEntry *entry, size_t alloc_nbytes, MemType free_mtype) {
    if (entry->nbytes <= alloc_nbytes) { return; }
#if 0
    if (mtype == MemType::kInfer) {
      auto insert_pos = entry->mem_entry_pos;
      ++insert_pos;
      auto *free_entry =
          CreateMemPoolEntry(entry->addr_offset + alloc_nbytes, entry->nbytes - alloc_nbytes,
                            free_mtype, insert_pos);
      freeblock_policy_->AddFreeBlock(free_entry);
      entry->nbytes = alloc_nbytes;
    } else { // kTrain
      auto insert_pos = entry->mem_entry_pos;
      auto free_entry =
          CreateMemPoolEntry(entry->addr_offset, entry->nbytes - alloc_nbytes,
                             free_mtype, insert_pos);
      freeblock_policy_->AddFreeBlock(free_entry);
      entry->addr_offset += entry->nbytes - alloc_nbytes;
      entry->nbytes = alloc_nbytes;
    }
#else
    auto insert_pos = entry->mem_entry_pos;
    ++insert_pos;
    auto *free_entry =
        CreateMemPoolEntry(entry->addr_offset + alloc_nbytes, entry->nbytes - alloc_nbytes,
                           free_mtype, insert_pos);
    freeblock_policy_->AddFreeBlock(free_entry);
    entry->nbytes = alloc_nbytes;
#endif
  };

  split_free_entry(entry, alloc_nbytes, free_mtype);
  CHECK(entry->mtype == MemType::kInfer || entry->mtype == MemType::kTrain);
  if (train_over_threshold && !alloc_by_merge) {
    FreeWithoutLock(entry); // free in local
    entry = freeblock_policy_->GetFreeBlock(nbytes, mtype, true, false);
    if (entry == nullptr) {
      DumpSummaryWithoutLock();
      LOG(FATAL) << "[mempool] " << detail::ByteDisplay(nbytes) << " should be find."; 
    }
    freeblock_policy_->RemoveFreeBlock(entry);
    auto free_mtype = entry->mtype;
    entry->mtype = mtype;
    stat_->at(static_cast<size_t>(mtype)).fetch_add(nbytes, std::memory_order_relaxed);
    stat_->at(static_cast<size_t>(free_mtype)).fetch_sub(nbytes, std::memory_order_relaxed);
    split_free_entry(entry, nbytes, free_mtype);
  }
  // auto t1 = std::chrono::steady_clock::now();
  // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
  // if (duration.count() > 50) {
  //   LOG(WARNING) << "[mempool] alloc " << mtype <<  " " << train_over_threshold << " "
  //                << detail::ByteDisplay(nbytes) << " " << duration.count() << " us.";
  // }

  return MakeSharedPtr(entry);
}
#endif

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
      minNbytes = std::min(minNbytes, entry->nbytes);
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
      {"train-local-free", MemType::kTrainLocalFree},
      {"infer", MemType::kInfer},
      {"train", MemType::kTrain}};
  auto usages = GetUsageWithoutLock();
  // std::cout << "now dump memory pool summary" << std::endl;
  LOG(INFO) << "---------- mempool summary ----------";
  for (auto&& [name, mtype] : mtype_list) {
    auto& usages_mtype = usages[mtype];
    LOG(INFO) << name << " stat:";
    LOG(INFO) << "\t max: "
              << detail::ByteDisplay(usages_mtype[UsageStat::kMaxNBytes]);
    LOG(INFO) << "\t min: "
              << detail::ByteDisplay(usages_mtype[UsageStat::kMinNBytes]);
    LOG(INFO) << "\t sum: "
              << detail::ByteDisplay(usages_mtype[UsageStat::kTotalNBytes]);
    LOG(INFO) << "\t cnt: "
              << usages_mtype[UsageStat::kCount];
    if (usages_mtype[UsageStat::kCount] == 0) { continue; }
    LOG(INFO) << "\t avg: "
              << detail::ByteDisplay(static_cast<size_t>(1.0 * usages_mtype[UsageStat::kTotalNBytes] /
                                     usages_mtype[UsageStat::kCount]));
  }
  LOG(INFO) << "Free: " << detail::ByteDisplay(GetMemUsage(MemType::kFree));
  LOG(INFO) << "TrainLocalFree: " << detail::ByteDisplay(GetMemUsage(MemType::kTrainLocalFree));

  LOG(INFO) << "Infer: " << detail::ByteDisplay(GetMemUsage(MemType::kInfer));
  LOG(INFO) << "Train: " << detail::ByteDisplay(GetMemUsage(MemType::kTrain));
  LOG(INFO) << "TrainAll: " << detail::ByteDisplay(GetMemUsage(MemType::kTrainAll)); 
  google::FlushLogFiles(google::INFO);
}

void MemPool::DumpBlockListWithoutLock() {
  // std::fstream stream{"mempool-blks", std::ios_base::out | std::ios_base::trunc};
  std::ofstream stream{"mempool-blks"};
  std::cout << "----- dump whole mempool (all) into ./mempool-blks -----" << std::endl;
  stream << "start,len,mtype,next,prev,mtype" << std::endl;
  for(auto it = mem_entry_list_->begin(); it != mem_entry_list_->end(); ++it) {
    auto *entry = GetEntry(segment_, *it);
    auto *prev = GetPrevEntry(segment_, entry, mem_entry_list_->begin());
    auto *next = GetNextEntry(segment_, entry, mem_entry_list_->end());
    stream << entry->addr_offset << "," 
           << entry->nbytes << ","
           << static_cast<int>(entry->mtype) << ","
           << (next ? next->addr_offset : -1) << "," 
           << (prev ? prev->addr_offset : -1) << ","
           << static_cast<unsigned>(entry->mtype) << std::endl;
  }
  stream.close();
}

void FreeListPolicy::InitMaster(MemPoolEntry *free_entry) {
  std::memcpy(policy_name_->data(), policy_name_str_.c_str(), policy_name_str_.size());
}

void FreeListPolicy::InitSlave() {
  CHECK_EQ(std::memcmp(policy_name_->data(), policy_name_str_.c_str(), policy_name_str_.size()), 0)
    << "policy name mismatch: " << policy_name_str_ << " vs " << policy_name_->data();
}

MemPoolEntry* BestFitPolicy::GetFreeBlock(size_t nbytes, MemType mtype, bool local, bool global) {
  CHECK(CheckGetFreeInput(nbytes, mtype));
  if (mtype == MemType::kTrain && local) {
    auto iter = free_entry_tables_[static_cast<size_t>(MemType::kTrainLocalFree)]
        ->lower_bound(nbytes);
    if (iter != free_entry_tables_[static_cast<size_t>(MemType::kTrainLocalFree)]->cend()) {
      return GetEntry(segment_, iter->second);
    }
  }
  if (global) {
    auto iter = free_entry_tables_[static_cast<size_t>(MemType::kFree)]
        ->lower_bound(nbytes);
    if (iter != free_entry_tables_[static_cast<size_t>(MemType::kFree)]->cend()) {
      return GetEntry(segment_, iter->second);
    }
  }
  return nullptr;
}

MemPoolEntry* BestFitPolicy::GetFreeBlockByMerge(size_t nbytes, MemType mtype, EntryList* mem_entry_list) {
  CHECK(mtype == MemType::kTrain);
  auto free_entry_table = free_entry_tables_[static_cast<size_t>(MemType::kTrainLocalFree)];
  for (auto it = free_entry_table->rbegin(); it != free_entry_table->rend(); it++) {
    auto* entry = GetEntry(segment_, it->second);
    auto prev_entry = GetPrevEntry(segment_, entry, mem_entry_list->begin());
    auto next_entry = GetNextEntry(segment_, entry, mem_entry_list->end());
    auto prev_free = prev_entry != nullptr && prev_entry->mtype == MemType::kFree;
    auto next_free = next_entry != nullptr && next_entry->mtype == MemType::kFree;
    auto prev_nbytes = prev_free ? prev_entry->nbytes : 0;
    auto next_nbytes = next_free ? next_entry->nbytes : 0;
    if (prev_nbytes + next_nbytes + entry->nbytes >= nbytes) {
      LOG(INFO) << " GetFreeBlockByMerge " << detail::ByteDisplay(prev_nbytes + next_nbytes + entry->nbytes)
                << " " << detail::ByteDisplay(nbytes);
      return entry;
    }
  }
  return nullptr;
}


void BestFitPolicy::NotifyUpdateFreeBlockNbytes(MemPoolEntry* entry,
                                                size_t old_nbytes) {
  CHECK(entry->IsFree()) << "update not free entry: " << *entry << " in free table";
  auto free_entry_table = free_entry_tables_[static_cast<size_t>(entry->mtype)];
  free_entry_table->erase(entry->freetable_pos);
  entry->freetable_pos = free_entry_table->insert(
      std::make_pair(entry->nbytes, GetHandle(segment_, entry)));
}

void BestFitPolicy::RemoveFreeBlock(MemPoolEntry* entry) {
  CHECK(entry->IsFree()) << "remove free entry: " << *entry << " from free table";
  auto free_entry_table = free_entry_tables_[static_cast<size_t>(entry->mtype)];
  free_entry_table->erase(entry->freetable_pos);
}

void BestFitPolicy::AddFreeBlock(MemPoolEntry* entry) {
  CHECK(entry->IsFree()) << "add not free entry: " << *entry << " to free table";
  auto free_entry_table = free_entry_tables_[static_cast<size_t>(entry->mtype)];
  entry->freetable_pos = free_entry_table->insert(
      std::make_pair(entry->nbytes, GetHandle(segment_, entry)));
}

void BestFitPolicy::RemoveLocalFreeBlocks(MemType mtype, std::function<void(MemPoolEntry*)> fn) {
  CHECK(mtype == MemType::kTrainLocalFree);
  auto local_free_entry_table = free_entry_tables_[static_cast<size_t>(mtype)];
  for (auto iter = local_free_entry_table->cbegin(); 
      iter != local_free_entry_table->cend(); iter++) {
    auto* entry = GetEntry(segment_, iter->second);
    fn(entry);
  }
  local_free_entry_table->clear();
}

void BestFitPolicy::CheckFreeList(const EntryListIterator& begin,
                                  const EntryListIterator& end) {
  DLOG(INFO) << "Check freelist";
  std::unordered_set<std::ptrdiff_t> free_set;
  for (size_t i = 0; i < static_cast<size_t>(MemType::kMemTypeFreeNum); i++) {
    auto free_entry_table = free_entry_tables_[i];
    for (auto iter = free_entry_table->cbegin(); iter != free_entry_table->cend();
        iter++) {
      auto* entry = GetEntry(segment_, iter->second);
      free_set.insert(entry->addr_offset);
      CHECK_EQ(entry->nbytes, iter->first)
          << "entry nbytes not match: " << *entry << ".";
      CHECK(entry->mtype == static_cast<MemType>(i))
          << "entry in freetable but not free: " << *entry << ".";
    }
  }

  for (auto iter = begin; iter != end; iter++) {
    auto* entry = GetEntry(segment_, *iter);
    if (entry->IsFree() &&
        free_set.find(entry->addr_offset) == free_set.cend()) {
      CHECK(entry->IsFree()) << "entry in free but not in free list: " << *entry << ".";
    }
  }
}

void BestFitPolicy::InitMaster(MemPoolEntry* free_entry) {
  FreeListPolicy::InitMaster(free_entry);
  AddFreeBlock(free_entry);
}

BestFitPolicy::BestFitPolicy(bip_shared_memory& segment)
    : FreeListPolicy(segment) {
  policy_name_str_ = "BestFitPolicy";
  for (size_t i = 0; i < static_cast<size_t>(MemType::kMemTypeFreeNum); i++) {
    auto free_table_name = "FreeTable-" + std::to_string(i);
    free_entry_tables_[i] = segment.find_or_construct<EntrySizeTable>(free_table_name.c_str())(
        segment_.get_segment_manager());
  }
}

void NextFitPolicy::InitMaster(MemPoolEntry *free_entry) {
  FreeListPolicy::InitMaster(free_entry);
  auto freelist = freelists_[static_cast<size_t>(free_entry->mtype)];
  free_entry->freelist_pos = freelist->insert(freelist->end(), GetHandle(segment_, free_entry));
}

void NextFitPolicy::NotifyUpdateFreeBlockNbytes(MemPoolEntry* entry,
                                                size_t old_nbytes) {
  CHECK(entry->IsFree())
      << "try update not free entry in freelist: " << *entry << ".";
}

void NextFitPolicy::AddFreeBlock(MemPoolEntry* entry) {
  CHECK(entry->IsFree())
      << "try add not free entry to freelist: " << *entry << ".";
  auto freelist = freelists_[static_cast<size_t>(entry->mtype)];
  entry->freelist_pos =
      freelist->insert(freelist->end(), GetHandle(segment_, entry));
}

void NextFitPolicy::RemoveFreeBlock(MemPoolEntry* entry) {
  CHECK(entry->IsFree())
      << "try to remove free entry from freelist: " << *entry << ".";
  auto freelist = freelists_[static_cast<size_t>(entry->mtype)];
  auto freelist_pos = freelist_poses_[static_cast<size_t>(entry->mtype)];
  if (entry->freelist_pos == *freelist_pos) {
    *freelist_pos = freelist->erase(entry->freelist_pos);
  } else {
    freelist->erase(entry->freelist_pos);
  }
}

void NextFitPolicy::CheckFreeList(const EntryListIterator& begin,
                                  const EntryListIterator& end) {
  DLOG(INFO) << "Check freelist";
  std::unordered_set<std::ptrdiff_t> free_set;
  for (size_t i = 0; i < static_cast<size_t>(MemType::kMemTypeFreeNum); i++) {
    auto freelist = freelists_[i];
    for (auto iter = freelist->cbegin(); iter != freelist->cend(); iter++) {
      auto* entry = GetEntry(segment_, *iter);
      free_set.insert(entry->addr_offset);
      CHECK(entry->mtype == static_cast<MemType>(i))
          << "entry in freelist but not free: " << *entry << ".";
    }
  }

  for (auto iter = begin; iter != end; iter++) {
    auto* entry = GetEntry(segment_, *iter);
    if (entry->IsFree() &&
        free_set.find(entry->addr_offset) == free_set.cend()) {
      CHECK(entry->IsFree())
          << "entry in free but not in free list: " << *entry << ".";
    }
  }
}

void NextFitPolicy::DumpFreeList(std::ostream& stream,
                                 const EntryListIterator& begin,
                                 const EntryListIterator& end) {
  stream << "start, len, allocated, next, prev, mtype" << std::endl;
  for (size_t i = 0; i < static_cast<size_t>(MemType::kMemTypeFreeNum); i++) {
    auto freelist = freelists_[i];
    for (const auto& element : *freelist) {
      auto* entry = GetEntry(segment_, element);
      auto* prev = GetPrevEntry(segment_, entry, begin);
      auto* next = GetNextEntry(segment_, entry, end);
      stream << entry->addr_offset << ", " << entry->nbytes << ", "
            << static_cast<int>(entry->mtype) << ", "
            << (prev ? next->addr_offset : -1) << ", "
            << (prev ? prev->addr_offset : -1) << ", "
            << static_cast<unsigned>(entry->mtype) << std::endl;
    }
  }
}

MemPoolEntry* FirstFitPolicy::GetFreeBlock(size_t nbytes, MemType mtype, bool local, bool global) {
  auto find_free_block = [this, nbytes, mtype] (EntryList* freelist) -> MemPoolEntry* {
    for (auto handle : *freelist) {
      auto* entry = GetEntry(this->segment_, handle);
      CHECK(entry->IsAvailableFree(mtype))
          << "not free entry in freelist: " << *entry << ".";
      if (entry->nbytes >= nbytes) {
        return entry;
      }
    }
    return nullptr;
  };

  if (mtype == MemType::kTrain && local) {
    auto* entry = find_free_block(
        freelists_[static_cast<size_t>(MemType::kTrainLocalFree)]);
    if (entry != nullptr) return entry;
  }
  if (!global) return nullptr;
  return find_free_block(freelists_[static_cast<size_t>(MemType::kFree)]);
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
