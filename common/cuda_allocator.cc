#include <common/log_as_glog_sta.h>
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
std::array<std::unique_ptr<CUDAMemPool>, MAX_DEVICE_NUM> CUDAMemPool::cuda_mem_pools_;

CUDAMemPool* CUDAMemPool::Get(int device_id) {
  auto ret = cuda_mem_pools_.at(device_id).get();
  CHECK(ret != nullptr) << "CUDAMemPool not initialized";
  return ret;
}

bool CUDAMemPool::IsEnable() {
  return allocate_tensor_from_memory_pool_;
}

void CUDAMemPool::Init(int device_id, std::size_t nbytes, 
                       bool cleanup, bool observe,
                       FreeListPolicyType free_list_policy,
                       bool enable_mpool) {
  CHECK(device_id < MAX_DEVICE_NUM) 
      << "device_id " << device_id << " exceeds MAX_DEVICE_NUM";

  allocate_tensor_from_memory_pool_ = enable_mpool;
  cuda_mem_pools_[device_id] = std::make_unique<CUDAMemPool>(
      device_id, nbytes, cleanup, observe, free_list_policy);
}

CUDAMemPool::CUDAMemPool(int device_id, std::size_t nbytes, 
                         bool cleanup, bool observe,
                         FreeListPolicyType free_list_policy) 
                         : device_id_{device_id} {
  if (!CUDAMemPool::IsEnable()) {
    return;
  }

  std::string prefix = GetDefaultShmNamePrefix(device_id);
  mpool::PagesPoolConf pages_pool_config{
      .device_id = device_id,
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

    // pages_pools_.push_back(pages_pool_);
    // torch_allocators_.push_back(torch_allocator);
    // tvm_allocators_.push_back(tvm_allocator);
  // }
}

std::shared_ptr<CUDAMemPool::PoolEntry>
CUDAMemPool::Alloc(size_t nbytes, MemType mtype, bool allow_nullptr) {
  return AllocWithStream(nbytes, mtype, 0, allow_nullptr);
}

std::shared_ptr<CUDAMemPool::PoolEntry> 
CUDAMemPool::AllocWithStream(std::size_t nbytes, MemType mtype, 
                             cudaStream_t stream, bool allow_nullptr) {
  CHECK(CUDAMemPool::IsEnable());
  CHECK(!allow_nullptr) << "currently deprecated";
  if (nbytes == 0) {
    return std::shared_ptr<CUDAMemPool::PoolEntry>{
        new PoolEntry{.addr = nullptr, .nbytes = nbytes, .mtype = mtype}};
  }

  std::unique_lock lock{mut_};
  mpool::MemBlock* mem_block;
  std::byte* ptr;
  if (mtype == MemType::kInfer) {
    mem_block = tvm_allocator_->GetObject()->Alloc(
        nbytes, 512, stream, 0);
    ptr = tvm_allocator_->GetObject()->GetBasePtr() 
          + mem_block->addr_offset;
  } else if (mtype == MemType::kTrain) {
    mem_block = torch_allocator_->GetObject()->Alloc(
        nbytes, 512, stream, mpool::VMMAllocator::ALLOC_TRY_EXPAND_VA);
    ptr = torch_allocator_->GetObject()->GetBasePtr() 
          + mem_block->addr_offset;
    DLOG(INFO) << "Torch Alloc: " << ptr 
               << " device " << device_id_ 
               << " nbytes " << nbytes;
  } else {
    LOG(FATAL) << "Unknown mtype: " << static_cast<size_t>(mtype);
  }

  auto free = [&, mtype](CUDAMemPool::PoolEntry* entry) {
    std::unique_lock lock{mut_};
    if (mtype == MemType::kInfer) {
      this->tvm_allocator_->GetObject()->Free(entry->block, 0);
    } else if (mtype == MemType::kTrain) {
      this->torch_allocator_->GetObject()->Free(entry->block, 0);
      DLOG(INFO) << "Torch Free: " << entry->addr
                 << ", device " << this->device_id_
                 << " nbytes = " << entry->nbytes;
    } else {
      LOG(FATAL) << "Unknown mtype: " << static_cast<size_t>(mtype);
    }
  };

  return std::shared_ptr<CUDAMemPool::PoolEntry>{
      new PoolEntry{.addr = ptr, 
                    .nbytes = nbytes, 
                    .mtype = mtype, 
                    .block = mem_block}, 
      free
    };
}

std::shared_ptr<CUDAMemPool::PoolEntry>
CUDAMemPool::RawAlloc(size_t nbytes, MemType mtype) {
  // static bool unified_memory = false;
  if (raw_alloc_enable_unified_memory_ == -1) {
    const char* env = getenv("STA_RAW_ALLOC_UNIFIED_MEMORY");
    if (env && atoi(env) != 0) {
      raw_alloc_enable_unified_memory_ = true;
      LOG(INFO) << "sta raw alloc using unified memory";
    } else {
      raw_alloc_enable_unified_memory_ = false;
    }
  }

  void *ptr;
  sta::DeviceGuard guard(device_id_);
  if (!raw_alloc_enable_unified_memory_) {
    COL_CUDA_CALL(cudaMalloc(&ptr, nbytes));
  } else {
    COL_CUDA_CALL(cudaMallocManaged(&ptr, nbytes));
  }
  return std::shared_ptr<PoolEntry>(
      new PoolEntry{ptr, nbytes, mtype}, [this](PoolEntry *entry) {
        COL_CUDA_CALL(cudaSetDevice(this->device_id_));
        COL_CUDA_CALL(cudaFree(entry->addr));
        delete entry;
      });
}

std::shared_ptr<CUDAMemPool::PoolEntry>
CUDAMemPool::HostAlloc(size_t nbytes, MemType mtype) {
  void *ptr;
  COL_CUDA_CALL(cudaMallocHost(&ptr, nbytes));
  return std::shared_ptr<PoolEntry>(
      new PoolEntry{ptr, nbytes, mtype}, [](PoolEntry *entry) {
        COL_CUDA_CALL(cudaFreeHost(entry->addr));
        delete entry;
      });
}

CUDAMemPool::~CUDAMemPool() {
  delete torch_allocator_;
  delete tvm_allocator_;
  delete pages_pool_;
}


size_t CUDAMemPool::InferMemUsage() {
  auto status = tvm_allocator_->GetObject()->GetStats();
  size_t nbytes =  status.mem_block_nbytes[false].allocated_free[0] 
                   + status.mem_block_nbytes[true].allocated_free[0];
  // LOG(INFO) << "InferMemUsage " << ByteDisplay(nbytes) << ".";
  return nbytes;
}

size_t CUDAMemPool::TrainMemUsage() {
  CHECK(CUDAMemPool::IsEnable());
  auto status = torch_allocator_->GetObject()->GetStats();
  size_t nbytes = status.mem_block_nbytes[false].allocated_free[0] 
                  + status.mem_block_nbytes[true].allocated_free[0];
  // LOG(INFO) << "TrainMemUsage " << ByteDisplay(nbytes) << ".";
  return nbytes;
}

size_t CUDAMemPool::TrainPeakMemUsage() {
  CHECK(CUDAMemPool::IsEnable());
  auto stats = torch_allocator_->GetObject()->GetStats();
  return stats.mem_block_nbytes[false].peak + stats.mem_block_nbytes[true].peak;
}

size_t CUDAMemPool::TrainAllMemUsage() {
  CHECK(CUDAMemPool::IsEnable());
  return (torch_allocator_->GetObject()->belong.GetPagesNum() 
          * pages_pool_->GetObject()->config.page_nbytes);
}

size_t CUDAMemPool::PoolNbytes() {
  CHECK(CUDAMemPool::IsEnable());
  return pages_pool_->GetObject()->config.pool_nbytes;
}

void CUDAMemPool::FreeTrainLocals() {
  CHECK(CUDAMemPool::IsEnable());
  torch_allocator_->GetObject()->EmptyCache();
}

void CUDAMemPool::DumpDumpBlockList() {}

void CUDAMemPool::RegisterOOMHandler(std::function<void()> oom_handler,
                                     MemType mtype) {
  CHECK(CUDAMemPool::IsEnable());
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

double CUDAMemPool::TrainAllocMs() { 
  LOG(FATAL) << "deprecated";
}

void CUDAMemPool::ResetTrainAllocMs() {
  LOG(FATAL) << "deprecated";
}

}  // namespace sta
}  // namespace colserve