#include <common/cuda_allocator.h>

#include <glog/logging.h>
#include <mpool/mapping_region.h>
#include <mpool/mem_block.h>
#include <mpool/caching_allocator.h>
#include <mpool/direct_allocator.h>
#include <mpool/pages.h>
#include <mpool/pages_pool.h>
#include <mpool/shm.h>
#include <mpool/vmm_allocator.h>


#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <common/util.h>
namespace colserve {
namespace sta {

std::unique_ptr<CUDAMemPool> CUDAMemPool::cuda_mem_pool_;

CUDAMemPool* CUDAMemPool::Get() {
  if (cuda_mem_pool_ == nullptr) {
    LOG(FATAL) << "[CUDAMemPool]: CUDAMemPool not initialized";
  }
  return cuda_mem_pool_.get();
}

void CUDAMemPool::Init(std::size_t nbytes, bool cleanup, bool observe,
                       FreeListPolicyType free_list_policy) {
  // DLOG(INFO) << "[CUDA Memory Pool] initilized with size " << size / 1024 / 1024 << " Mb";
  cuda_mem_pool_ =
      std::make_unique<CUDAMemPool>(nbytes, cleanup, observe, free_list_policy);
}

CUDAMemPool::CUDAMemPool(std::size_t nbytes, bool cleanup, bool observe,
                         FreeListPolicyType free_list_policy) {
  //    remove("/dev/shm/gpu_colocation_mempool");
  std::string prefix =
      "gpu_colocation_" + std::string(getenv("CUDA_VISIBLE_DEVICES"));
  mpool::PagesPoolConf pages_pool_config{
      .device_id = 0,
      .page_nbytes = 32_MB,
      .pool_nbytes = nbytes,
      .shm_name = prefix + "_pages_pool",
      .log_prefix = "[PagesPool] ",
      .shm_nbytes = 4_MB,
  };
  mpool::VMMAllocatorConfig torch_allocator_config{
      .log_prefix = "[TorchAllocator] ",
      .shm_name = prefix + "_torch_allocator",
      .shm_nbytes = 64_MB,
      .va_range_scale = 8,
      .belong_name = "Torch",
      .small_block_nbytes = 2_MB,
      .align_nbytes = 512};
  mpool::VMMAllocatorConfig tvm_allocator_config{
      .log_prefix = "[TVMAllocator] ",
      .shm_name = prefix + "_tvm_allocator",
      .shm_nbytes = 64_MB,
      .va_range_scale = 1 ,
      .belong_name = "TVM",
      .small_block_nbytes = 32_MB,
      .align_nbytes = 512};
  if (cleanup) {
    mpool::PagesPool::RemoveShm(pages_pool_config);
    mpool::VMMAllocator::RemoveShm(torch_allocator_config);
    mpool::VMMAllocator::RemoveShm(tvm_allocator_config);
  }
  pages_pool_ = new mpool::SharableObject<mpool::PagesPool>{
      pages_pool_config.shm_name, pages_pool_config.shm_nbytes,
      pages_pool_config};
  torch_allocator_ = new mpool::SharableObject<mpool::CachingAllocator>{
      torch_allocator_config.shm_name, torch_allocator_config.shm_nbytes,
      *pages_pool_->GetObject(), torch_allocator_config};
  tvm_allocator_ = new mpool::SharableObject<mpool::DirectAllocator>{
      tvm_allocator_config.shm_name, tvm_allocator_config.shm_nbytes,
      *pages_pool_->GetObject(), tvm_allocator_config};
}

std::shared_ptr<CUDAMemPool::PoolEntry> CUDAMemPool::Alloc(std::size_t nbytes,
                                                           MemType mtype,
                                                           bool allow_nullptr) {
  CHECK(!allow_nullptr) << "currently deprecated";
  if (nbytes == 0) {
    return std::shared_ptr<CUDAMemPool::PoolEntry>{
        new PoolEntry{.addr = nullptr, .nbytes = nbytes, .mtype = mtype}};
  }

  static std::mutex mutex_;
  std::unique_lock lock{mutex_};
  auto t0 = std::chrono::steady_clock::now();
  mpool::MemBlock* mem_block;
  std::byte* ptr;
  if (mtype == MemType::kInfer) {
    mem_block = tvm_allocator_->GetObject()->Alloc(nbytes, 512, 0, 0);
    ptr = tvm_allocator_->GetObject()->GetBasePtr() + mem_block->addr_offset;
  } else if (mtype == MemType::kTrain) {
    mem_block = torch_allocator_->GetObject()->Alloc(nbytes, 512, 0, mpool::VMMAllocator::ALLOC_TRY_EXPAND_VA);
    ptr = torch_allocator_->GetObject()->GetBasePtr() + mem_block->addr_offset;
    DLOG(INFO) << "Torch Alloc: " << ptr << ", nbytes = " << nbytes;
  } else {
    LOG(FATAL) << "Unknown mtype: " << static_cast<size_t>(mtype);
  }
  auto t1 = std::chrono::steady_clock::now();
  if (mtype == MemType::kTrain) {
    train_alloc_us_.fetch_add(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(),
        std::memory_order_relaxed);
  }
  // DLOG(INFO) << "mtype = " << static_cast<size_t>(mtype) << ", alloc time = "
  //            << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() << ".";

  auto free = [&, mtype](CUDAMemPool::PoolEntry* entry) {
    std::unique_lock lock{mutex_};
    if (mtype == MemType::kInfer) {
      tvm_allocator_->GetObject()->Free(entry->block, 0);
    } else if (mtype == MemType::kTrain) {
      torch_allocator_->GetObject()->Free(entry->block, 0);
      DLOG(INFO) << "Torch Free: " << entry->addr
                 << ", nbytes = " << entry->nbytes;
    } else {
      LOG(FATAL) << "Unknown mtype: " << static_cast<size_t>(mtype);
    }
  };

  return std::shared_ptr<CUDAMemPool::PoolEntry>{
      new PoolEntry{.addr = ptr, .nbytes = nbytes, .mtype = mtype, .block = mem_block}, free};
}

std::shared_ptr<CUDAMemPool::PoolEntry> CUDAMemPool::RawAlloc(size_t nbytes,
                                                              MemType mtype) {
  static bool initilized = false;
  static bool unified_memory = false;
  if (!initilized) {
    const char* env = getenv("STA_RAW_ALLOC_UNIFIED_MEMORY");
    if (env && atoi(env) != 0) {
      unified_memory = true;
      LOG(INFO) << "sta raw alloc using unified memory";
    }
    initilized = true;
  }

  void* ptr;
  CUDA_CALL(cudaSetDevice(0));
  if (!unified_memory) {
    CUDA_CALL(cudaMalloc(&ptr, nbytes));
  } else {
    CUDA_CALL(cudaMallocManaged(&ptr, nbytes));
  }
  return std::shared_ptr<PoolEntry>(new PoolEntry{ptr, nbytes, mtype},
                                    [](PoolEntry* entry) {
                                      CUDA_CALL(cudaSetDevice(0));
                                      CUDA_CALL(cudaFree(entry->addr));
                                      delete entry;
                                    });
}

CUDAMemPool::~CUDAMemPool() {
  delete torch_allocator_;
  delete tvm_allocator_;
  delete pages_pool_;
}

// void CUDAMemPoolImpl::DumpSummary() {
//   DLOG(INFO) << "---------- mempool summary ----------";
//   DLOG(INFO) << "free blocks: " << size2entry_->size();
//   DLOG(INFO) << "free size: " << std::accumulate(size2entry_->cbegin(), size2entry_->cend(), 0L,
//                                                 [](auto acc, auto &&pair) { return acc + pair.first; });
//   DLOG(INFO) << "largest free block size: " << (--size2entry_->cend())->first;
//   DLOG(INFO) << "total blocks: " << addr2entry_->size();
//   DLOG(INFO) << "total size: " << std::accumulate(addr2entry_->cbegin(), addr2entry_->cend(), 0L,
//                                                  [&](auto acc, auto &&pair) {
//                                                    return acc + GetEntry(pair.second)->nbytes;
//                                                  });
//   DLOG(INFO) << "infer usage: " << InferMemUsage();
//   DLOG(INFO) << "train usage: " << TrainMemUsage();
// }

// bool CUDAMemPoolImpl::CheckAddr(void *addr) {
//   bool ok = reinterpret_cast<size_t>(addr) >= reinterpret_cast<size_t>(mem_pool_base_ptr_)
//       && reinterpret_cast<size_t>(addr) <= reinterpret_cast<size_t>(static_cast<char*>(mem_pool_base_ptr_) + config_.cuda_memory_size);
//   if (!ok) {
//     LOG(WARNING) << "MemoryPool CheckAddr " << addr << " failed"
//                  << " memory range [" << mem_pool_base_ptr_ << ", "
//                  << static_cast<void*>(static_cast<char*>(mem_pool_base_ptr_) + config_.cuda_memory_size) << "]";
//   }
//   return ok;
// }

size_t CUDAMemPool::InferMemUsage() {
  auto status = Get()->tvm_allocator_->GetObject()->GetStats();
  size_t nbytes =  status.mem_block_nbytes[false].allocated_free[0] + status.mem_block_nbytes[true].allocated_free[0];
  // LOG(INFO) << "InferMemUsage " << ByteDisplay(nbytes) << ".";
  return nbytes;
}

size_t CUDAMemPool::TrainMemUsage() {
  auto status = Get()->torch_allocator_->GetObject()->GetStats();
  size_t nbytes = status.mem_block_nbytes[false].allocated_free[0] + status.mem_block_nbytes[true].allocated_free[0];
  // LOG(INFO) << "TrainMemUsage " << ByteDisplay(nbytes) << ".";
  return nbytes;
}

size_t CUDAMemPool::TrainPeakMemUsage() {
  auto stats = Get()->torch_allocator_->GetObject()->GetStats();
  return stats.mem_block_nbytes[false].peak + stats.mem_block_nbytes[true].peak;
}

size_t CUDAMemPool::TrainAllMemUsage() {
  // return TorchAllocator::Get().PeekAllocatedNbytes();
  // return MemPool::Get().GetPhyMemPageNbytes(Belong::kTrain);
  return Get()->torch_allocator_->GetObject()->belong.GetAllocatedNbytes();
}

size_t CUDAMemPool::PoolNbytes() {
  return Get()->pages_pool_->GetObject()->config.pool_nbytes;
}

void CUDAMemPool::FreeTrainLocals() {
  Get()->torch_allocator_->GetObject()->EmptyCache();
}

void CUDAMemPool::DumpDumpBlockList() {}

void CUDAMemPool::RegisterOOMHandler(std::function<void()> oom_handler,
                                     MemType mtype) {
  auto oom_observer = std::make_shared<mpool::OOMObserver>(
      [oom_handler](int device_id, cudaStream_t stream, OOMReason reason) {
        oom_handler();
      });
  switch (mtype) {
    case MemType::kInfer:
      tvm_allocator_->GetObject()->AddOOMObserver(oom_observer);
      break;
    case MemType::kTrain:
      torch_allocator_->GetObject()->AddOOMObserver(oom_observer);
      break;
    default:
      LOG(FATAL) << "unknown MemType " << static_cast<int>(mtype) << ".";
  }
}

}  // namespace sta
}  // namespace colserve