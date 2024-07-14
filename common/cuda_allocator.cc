#include "log_as_glog_sta.h"
#include <common/cuda_allocator.h>
#include <common/util.h>
#include <common/device_manager.h>
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

namespace colserve {
namespace sta {

bool CUDAMemPool::allocate_tensor_from_memory_pool_ = false;
std::unique_ptr<CUDAMemPool> CUDAMemPool::cuda_mem_pool_;

CUDAMemPool* CUDAMemPool::Get() {
  if (cuda_mem_pool_ == nullptr) {
    LOG(FATAL) << "[CUDAMemPool]: CUDAMemPool not initialized";
  }
  return cuda_mem_pool_.get();
}

bool CUDAMemPool::IsEnable() {
  return allocate_tensor_from_memory_pool_;
}

void CUDAMemPool::Init(std::size_t nbytes, bool cleanup, bool observe,
                       FreeListPolicyType free_list_policy) {
  // DLOG(INFO) << "[CUDA Memory Pool] initilized with size " << size / 1024 / 1024 << " Mb";
  cuda_mem_pool_ = std::make_unique<CUDAMemPool>(nbytes, cleanup, 
                                                 observe, free_list_policy);
  allocate_tensor_from_memory_pool_ = true;
}

CUDAMemPool::CUDAMemPool(std::size_t nbytes, bool cleanup, bool observe,
                         FreeListPolicyType free_list_policy) {
  for (int i = 0; i < DeviceManager::GetNumVisibleGpu(); i++) {
    std::string prefix = GetDefaultShmNamePrefix(i);
    mpool::PagesPoolConf pages_pool_config{
        .device_id = i,
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
    auto pages_pool_ = new mpool::SharableObject<mpool::PagesPool>{
        pages_pool_config.shm_name, pages_pool_config.shm_nbytes,
        pages_pool_config};
    auto torch_allocator = new mpool::SharableObject<mpool::CachingAllocator>{
        torch_allocator_config.shm_name, torch_allocator_config.shm_nbytes,
        *pages_pool_->GetObject(), torch_allocator_config};
    auto tvm_allocator = new mpool::SharableObject<mpool::DirectAllocator>{
        tvm_allocator_config.shm_name, tvm_allocator_config.shm_nbytes,
        *pages_pool_->GetObject(), tvm_allocator_config};

    pages_pools_.push_back(pages_pool_);
    torch_allocators_.push_back(torch_allocator);
    tvm_allocators_.push_back(tvm_allocator);
  }
}

std::shared_ptr<CUDAMemPool::PoolEntry> 
CUDAMemPool::Alloc(int device_id, std::size_t nbytes, 
                   MemType mtype, bool allow_nullptr) {
  CHECK(!allow_nullptr) << "currently deprecated";
  if (nbytes == 0) {
    return std::shared_ptr<CUDAMemPool::PoolEntry>{
        new PoolEntry{.addr = nullptr, .nbytes = nbytes, .mtype = mtype}};
  }

  static std::mutex mutex_;
  std::unique_lock lock{mutex_};
  // auto t0 = std::chrono::steady_clock::now();
  mpool::MemBlock* mem_block;
  std::byte* ptr;
  if (mtype == MemType::kInfer) {
    mem_block = cuda_mem_pool_->tvm_allocators_[device_id]->GetObject()->Alloc(
        nbytes, 512, 0, 0);
    ptr = cuda_mem_pool_->tvm_allocators_[device_id]->GetObject()->GetBasePtr() 
          + mem_block->addr_offset;
  } else if (mtype == MemType::kTrain) {
    mem_block = cuda_mem_pool_->torch_allocators_[device_id]->GetObject()->Alloc(
        nbytes, 512, 0, mpool::VMMAllocator::ALLOC_TRY_EXPAND_VA);
    ptr = cuda_mem_pool_->torch_allocators_[device_id]->GetObject()->GetBasePtr() 
          + mem_block->addr_offset;
    DLOG(INFO) << "Torch Alloc: " << ptr << ", nbytes = " << nbytes;
  } else {
    LOG(FATAL) << "Unknown mtype: " << static_cast<size_t>(mtype);
  }

  auto free = [&, mtype, device_id](CUDAMemPool::PoolEntry* entry) {
    std::unique_lock lock{mutex_};
    if (mtype == MemType::kInfer) {
      Get()->tvm_allocators_[device_id]->GetObject()->Free(entry->block, 0);
    } else if (mtype == MemType::kTrain) {
      Get()->torch_allocators_[device_id]->GetObject()->Free(entry->block, 0);
      DLOG(INFO) << "Torch Free: " << entry->addr
                 << ", nbytes = " << entry->nbytes;
    } else {
      LOG(FATAL) << "Unknown mtype: " << static_cast<size_t>(mtype);
    }
  };

  return std::shared_ptr<CUDAMemPool::PoolEntry>{
      new PoolEntry{.addr = ptr, .nbytes = nbytes, .mtype = mtype, .block = mem_block}, 
      free
    };
}

std::shared_ptr<CUDAMemPool::PoolEntry>
CUDAMemPool::RawAlloc(int device_id, size_t nbytes, MemType mtype) {
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

  void *ptr;
  CUDA_CALL(cudaSetDevice(device_id));
  if (!unified_memory) {
    CUDA_CALL(cudaMalloc(&ptr, nbytes));
  } else {
    CUDA_CALL(cudaMallocManaged(&ptr, nbytes));
  }
  return std::shared_ptr<PoolEntry>(
      new PoolEntry{ptr, nbytes, mtype}, [device_id](PoolEntry *entry) {
        CUDA_CALL(cudaSetDevice(device_id));
        CUDA_CALL(cudaFree(entry->addr));
        delete entry;
      });
}

CUDAMemPool::~CUDAMemPool() {
  for (auto p : torch_allocators_) delete p;
  for (auto p : tvm_allocators_) delete p;
  for (auto p : pages_pools_) delete p;

  torch_allocators_.clear();
  tvm_allocators_.clear();
  pages_pools_.clear();
}


size_t CUDAMemPool::InferMemUsage(int device_id) {
  auto status = Get()->tvm_allocators_[device_id]->GetObject()->GetStats();
  size_t nbytes =  status.mem_block_nbytes[false].allocated_free[0] 
                   + status.mem_block_nbytes[true].allocated_free[0];
  // LOG(INFO) << "InferMemUsage " << ByteDisplay(nbytes) << ".";
  return nbytes;
}

size_t CUDAMemPool::TrainMemUsage(int device_id) {
  auto status = Get()->torch_allocators_[device_id]->GetObject()->GetStats();
  size_t nbytes = status.mem_block_nbytes[false].allocated_free[0] 
                  + status.mem_block_nbytes[true].allocated_free[0];
  // LOG(INFO) << "TrainMemUsage " << ByteDisplay(nbytes) << ".";
  return nbytes;
}

size_t CUDAMemPool::TrainPeakMemUsage(int device_id) {
  auto stats = Get()->torch_allocators_[device_id]->GetObject()->GetStats();
  return stats.mem_block_nbytes[false].peak + stats.mem_block_nbytes[true].peak;
}

size_t CUDAMemPool::TrainAllMemUsage(int device_id) {
  return (Get()->torch_allocators_[device_id]->GetObject()->belong.GetPagesNum() 
          * Get()->pages_pools_[device_id]->GetObject()->config.page_nbytes);
}

size_t CUDAMemPool::PoolNbytes(int device_id) {
  return Get()->pages_pools_[device_id]->GetObject()->config.pool_nbytes;
}

void CUDAMemPool::FreeTrainLocals(int device_id) {
  Get()->torch_allocators_[device_id]->GetObject()->EmptyCache();
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
      for (auto allocator : tvm_allocators_) {
        allocator->GetObject()->AddOOMObserver(oom_observer);
      }
      break;
    case MemType::kTrain:
      for (auto allocator : torch_allocators_) {
        allocator->GetObject()->AddOOMObserver(oom_observer);
      }
      break;
    default:
      LOG(FATAL) << "unknown MemType " << static_cast<int>(mtype) << ".";
  }
}

double CUDAMemPool::TrainAllocMs() { 
  LOG(FATAL) << "deprecated";
}

void CUDAMemPool::ResetTrainAllocMs() {
  LOG(FATAL) << "deprecated";
}

}  // namespace sta
}  // namespace colserve