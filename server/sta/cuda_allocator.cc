#include <iostream>
#include <cuda_runtime_api.h>
// #include <glog/logging.h>

#include "cuda_allocator.h"

namespace colserve {
namespace sta {

#define CUDA_CALL(func) do { \
    auto error = func; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error); \
      exit(EXIT_FAILURE); \
    } \
  } while (0);

std::unique_ptr<CUDAMemPool> CUDAMemPool::cuda_mem_pool_;

CUDAMemPool* CUDAMemPool::Get() {
  return cuda_mem_pool_.get();
}

void CUDAMemPool::Init(std::size_t size) {
  // LOG(INFO) << "[CUDA Memory Pool] initilized with size " << size / 1024 / 1024 << " Mb";
  cuda_mem_pool_ = std::make_unique<CUDAMemPool>(size);
}

CUDAMemPool::CUDAMemPool(std::size_t size) : entry_by_addr_{CmpPoolEntryByAddr} {
  CUDA_CALL(cudaSetDevice(0));
  auto entry = PoolEntry{nullptr, size};
  CUDA_CALL(cudaMalloc(&entry.addr, size));
  
  std::unique_lock<std::mutex> lock{mutex_};
  entry_by_addr_.insert(entry);
} 

CUDAMemPool::PoolEntry CUDAMemPool::Alloc(std::size_t size) {
  std::unique_lock lock{mutex_};
  // auto it = std::lower_bound(entry_by_addr_.begin(), entry_by_addr_.end(), size, 
  //   [](const std::shared_ptr<PoolEntry> &a, const std::size_t &b) -> bool
  //   {return a->size < b;});
  auto it = entry_by_addr_.begin();
  for (; it != entry_by_addr_.end(); it++) {
    if (it->size >= size) break;
  }
  if (it != entry_by_addr_.end()) {
    auto entry = *it;
    entry_by_addr_.erase(it);
    
    std::size_t aligned_size = (size + 1023) & ~1023;
    if (entry.size > aligned_size) {
      auto new_entry = PoolEntry{
          (void*)((std::size_t)entry.addr + aligned_size), 
          entry.size - aligned_size
      };
      entry_by_addr_.insert(new_entry);
      entry.size = aligned_size;
    }
    return PoolEntry{entry.addr, entry.size};
  } else {
    return PoolEntry{};
  }
}

void CUDAMemPool::Free(PoolEntry entry) {
  std::unique_lock lock{mutex_};
  auto it = entry_by_addr_.lower_bound(entry);
  if (it != entry_by_addr_.begin()) {
    auto prev_it = it;
    prev_it--;
    auto prev_end = (std::size_t)prev_it->addr + prev_it->size;
    if (prev_end == (size_t)entry.addr) {
      entry.addr = prev_it->addr;
      entry.size += prev_it->size;
      entry_by_addr_.erase(prev_it);
    }
  }
  if (it != entry_by_addr_.end()) {
    auto next_it = it;
    auto next_begin = (std::size_t)next_it->addr;
    if (next_begin == (std::size_t)entry.addr + entry.size) {
      entry.size += next_it->size;
      entry_by_addr_.erase(next_it);
    }
  }
  entry_by_addr_.insert(entry);
}

bool CUDAMemPool::CmpPoolEntryByAddr(const PoolEntry &a, const PoolEntry &b) {
  return a.addr < b.addr;
}



}
}