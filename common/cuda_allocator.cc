#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>

#include "tvm_allocator.h"
#include "cuda_allocator.h"
#include "mempool.h"
#include "torch_allocator.h"
#include <glog/logging.h>
#include <memory>
#include <numeric>
#include <string>

#define CUDA_CALL(func) do { \
  auto error = func; \
  if (error != cudaSuccess) { \
    LOG(FATAL) << #func << " " << cudaGetErrorString(error); \
    exit(EXIT_FAILURE); \
  } \
  } while (0)


namespace colserve {
namespace sta {

std::unique_ptr<CUDAMemPool> CUDAMemPool::cuda_mem_pool_;

CUDAMemPool *CUDAMemPool::Get() {
  if (cuda_mem_pool_ == nullptr) {
    LOG(FATAL) << "[CUDAMemPool]: CUDAMemPool not initialized";
  }
  return cuda_mem_pool_.get();
}

void CUDAMemPool::Init(std::size_t nbytes, bool cleanup, bool observe, FreeListPolicyType free_list_policy) {
  // DLOG(INFO) << "[CUDA Memory Pool] initilized with size " << size / 1024 / 1024 << " Mb";
  cuda_mem_pool_ = std::make_unique<CUDAMemPool>(nbytes, cleanup, observe, free_list_policy);
}

CUDAMemPool::CUDAMemPool(std::size_t nbytes, bool cleanup, bool observe, FreeListPolicyType free_list_policy) {
//    remove("/dev/shm/gpu_colocation_mempool");
  std::string nbytes_s = std::to_string(nbytes);
  std::string cleanup_s = cleanup ? "1" : "0";
  CHECK_EQ(setenv("COL_MEMPOOL_NBYTES", nbytes_s.c_str(), true), 0);
  CHECK_EQ(setenv("COL_MEMPOOL_CLEANUP", cleanup_s.c_str(), true), 0);
  TorchAllocator::Get();
  
}

static std::set<void *> train_set;

std::shared_ptr<CUDAMemPool::PoolEntry> CUDAMemPool::Alloc(
    std::size_t nbytes, MemType mtype, bool allow_nullptr) {
  static std::mutex mutex_;
  std::unique_lock lock{mutex_};
  auto t0 = std::chrono::steady_clock::now();
  std::byte *ptr;
  if (mtype == MemType::kInfer) {
    ptr = TVMAllocator::Get().Alloc(nbytes, true);
  } else if (mtype == MemType::kTrain) {
    ptr = TorchAllocator::Get().Alloc(nbytes, true);
    if (ptr != nullptr) {
      CHECK(train_set.insert(ptr).second == true) << ptr;
    }

    DLOG(INFO) << "Torch Alloc: " << ptr << ", nbytes = " << nbytes;
  } else {
    LOG(FATAL) << "Unknown mtype: " << static_cast<size_t>(mtype); 
  }
  auto t1 = std::chrono::steady_clock::now();
  if (mtype == MemType::kTrain) {
    train_alloc_us_.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
  }
  // DLOG(INFO) << "mtype = " << static_cast<size_t>(mtype) << ", alloc time = " << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() << ".";
  auto free = [mtype](CUDAMemPool::PoolEntry *entry) {
      std::unique_lock lock{mutex_};
    if (mtype == MemType::kInfer) {
      TVMAllocator::Get().Free(reinterpret_cast<std::byte*>(entry->addr));


    } else if (mtype == MemType::kTrain) {
      TorchAllocator::Get().Free(reinterpret_cast<std::byte*>(entry->addr));
      DLOG(INFO) << "Torch Free: " << entry->addr << ", nbytes = " << entry->nbytes;
      train_set.erase(entry->addr);
    } else {
      LOG(FATAL) << static_cast<int>(mtype);
    }

  };
  return std::shared_ptr<CUDAMemPool::PoolEntry>{
    new PoolEntry{.addr = ptr, .nbytes=nbytes, .mtype = mtype}, free
  };
}

std::shared_ptr<CUDAMemPool::PoolEntry> CUDAMemPool::RawAlloc(size_t nbytes, MemType mtype) {
  static bool initilized = false;
  static bool unified_memory = false;
  if (!initilized) {
    const char* env = getenv("STA_RAW_ALLOC_UNIFIED_MEMORY");
    if (env && atoi(env) != 0) {
      unified_memory = true;
      DLOG(INFO) << "sta raw alloc using unified memory";

    }
    initilized = true;
  }

  void *ptr;
  CUDA_CALL(cudaSetDevice(0));
  if (!unified_memory) {
    CUDA_CALL(cudaMalloc(&ptr, nbytes));
  } else {
    CUDA_CALL(cudaMallocManaged(&ptr, nbytes));
  }
  return std::shared_ptr<PoolEntry>(
      new PoolEntry{ptr, nbytes, mtype}, [](PoolEntry *entry) {
        CUDA_CALL(cudaSetDevice(0));
        CUDA_CALL(cudaFree(entry->addr));
        delete entry;
      });
}

CUDAMemPool::~CUDAMemPool() {

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
 return MemPool::Get().GetAllocatedNbytes(Belong::kInfer);
}

size_t CUDAMemPool::TrainMemUsage() {
 return MemPool::Get().GetAllocatedNbytes(Belong::kTrain);
}

size_t CUDAMemPool::TrainAllMemUsage() {
  return MemPool::Get().GetCachedNbytes(Belong::kTrain);
}

size_t CUDAMemPool::PoolNbytes() {
  return MemPool::Get().mempool_nbytes;
}

void CUDAMemPool::FreeTrainLocals() {
  TorchAllocator::Get().EmptyCache();
}

void CUDAMemPool::DumpDumpBlockList() {

}

void CUDAMemPool::RegisterOOMHandler(std::function<void()> oom_handler, MemType mtype) {
  switch (mtype) {
    case MemType::kInfer:
      TVMAllocator::Get().RegisterOOMHandler(oom_handler);
      break;
    case MemType::kTrain:
      TorchAllocator::Get().RegisterOOMHandler(oom_handler);
      break;
    default:
      LOG(FATAL) << "unknown MemType " << static_cast<int>(mtype) << ".";
  }
}
}  // namespace sta
}