#include <cstdint>
#include <exception>
#include <iostream>

#include "cuda_allocator.h"
#include "sta/mempool.h"
#include <glog/logging.h>
#include <numeric>

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

void CUDAMemPool::Init(std::size_t nbytes, bool master) {
  // LOG(INFO) << "[CUDA Memory Pool] initilized with size " << size / 1024 / 1024 << " Mb";
  cuda_mem_pool_ = std::make_unique<CUDAMemPool>(nbytes, master);
}





CUDAMemPool::CUDAMemPool(std::size_t nbytes, bool master) {
//    remove("/dev/shm/gpu_colocation_mempool");
  MemPoolConfig config = GetDefaultMemPoolConfig(nbytes);
  impl_ = new MemPool{config, master, false};
}

std::shared_ptr<CUDAMemPool::PoolEntry> CUDAMemPool::Alloc(
    std::size_t nbytes, MemType mtype, bool allow_nullptr) {
  auto ret = impl_->Alloc(nbytes, mtype);
  if (!allow_nullptr && ret == nullptr) {
    impl_->DumpSummary();
    LOG(FATAL) << "request size " << nbytes << " byte ( " << detail::ByteToMB(nbytes) << " mb )"  << " out of free gpu memory";
  }
  return ret;
}

std::shared_ptr<CUDAMemPool::PoolEntry> CUDAMemPool::Resize(
    std::shared_ptr<PoolEntry> entry, std::size_t nbytes) {
  // TODO: handle reallocate
  CHECK(entry != nullptr);
  auto ptr = impl_->Alloc(nbytes, entry->mtype);
  CopyFromTo(entry, ptr);
  return ptr;
}

void CUDAMemPool::CopyFromTo(std::shared_ptr<PoolEntry> src, std::shared_ptr<PoolEntry> dst) {
  impl_->CopyFromTo(src, dst);
}


std::shared_ptr<CUDAMemPool::PoolEntry> CUDAMemPool::RawAlloc(size_t nbytes, MemType mtype) {
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
  delete impl_;
  impl_ = nullptr;
}



// void CUDAMemPoolImpl::DumpSummary() {
//   LOG(INFO) << "---------- mempool summary ----------";
//   LOG(INFO) << "free blocks: " << size2entry_->size();
//   LOG(INFO) << "free size: " << std::accumulate(size2entry_->cbegin(), size2entry_->cend(), 0L,
//                                                 [](auto acc, auto &&pair) { return acc + pair.first; });
//   LOG(INFO) << "largest free block size: " << (--size2entry_->cend())->first;
//   LOG(INFO) << "total blocks: " << addr2entry_->size();
//   LOG(INFO) << "total size: " << std::accumulate(addr2entry_->cbegin(), addr2entry_->cend(), 0L,
//                                                  [&](auto acc, auto &&pair) {
//                                                    return acc + GetEntry(pair.second)->nbytes;
//                                                  });
//   LOG(INFO) << "infer usage: " << InferMemUsage();
//   LOG(INFO) << "train usage: " << TrainMemUsage();
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
  return Get()->impl_->InferMemUsage();
}

size_t CUDAMemPool::TrainMemUsage() {
  return Get()->impl_->TrainMemUsage();
}

void CUDAMemPool::ReleaseMempool() {}

size_t CUDAMemPool::PoolNbytes() {
  CHECK(cuda_mem_pool_ != nullptr);
  return Get()->impl_->PoolNbytes();
}
}  // namespace sta
}