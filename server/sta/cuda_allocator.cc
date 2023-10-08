#include <iostream>
#include <cuda_runtime_api.h>

#include "cuda_allocator.h"
#include <glog/logging.h>

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
  CUDA_CALL(cudaStreamCreate(&stream_));

  auto entry = PoolEntry{nullptr, size};
  CUDA_CALL(cudaMalloc(&entry.addr, size));
  
  std::unique_lock<std::mutex> lock{mutex_};
  entry_by_addr_.insert(entry);
}


std::shared_ptr<CUDAMemPool::PoolEntry> CUDAMemPool::Alloc(std::size_t size) {
  if (size == 0) {
    return nullptr;
  }
  std::unique_lock lock{mutex_};
  return AllocUnCheckUnlock(size);
}

std::shared_ptr<CUDAMemPool::PoolEntry> CUDAMemPool::Resize(
    std::shared_ptr<PoolEntry> entry, std::size_t size) {
  if (size == 0) return nullptr;
  if (entry != nullptr && entry->size >= size) {
    return entry;
  }
  std::unique_lock lock{mutex_};
  // case 1: extend the entry by merge with the next entry
  
  // case 2: alloc new entry
  auto new_entry = AllocUnCheckUnlock(size);
  CHECK(new_entry != nullptr) << "CUDAMemPool: out of memory";
  return new_entry;
}

void CUDAMemPool::CopyFromTo(std::shared_ptr<PoolEntry> src, std::shared_ptr<PoolEntry> dst) {
  if (src == nullptr || dst == nullptr) return;
  if (src->addr == dst->addr) return;

  CUDA_CALL(cudaMemcpyAsync(dst->addr, src->addr, 
      std::min(src->size, dst->size), cudaMemcpyDefault, stream_));
  CUDA_CALL(cudaStreamSynchronize(stream_));
}

std::shared_ptr<CUDAMemPool::PoolEntry> CUDAMemPool::AllocUnCheckUnlock(std::size_t size) {
  auto it = entry_by_addr_.begin();
  for (; it != entry_by_addr_.end(); it++) {
    if (it->size >= size) break;
  }
  if (it != entry_by_addr_.end()) {
    auto entry = *it;
    entry_by_addr_.erase(it);
    
    size_t aligned_size = AlignSize(size);
    if (entry.size > aligned_size) {
      auto new_entry = PoolEntry{
          (void*)((std::size_t)entry.addr + aligned_size), 
          entry.size - aligned_size
      };
      entry_by_addr_.insert(new_entry);
      entry.size = aligned_size;
    }
    // return PoolEntry{entry.addr, entry.size};
    return std::shared_ptr<CUDAMemPool::PoolEntry>(
        new PoolEntry{entry.addr, entry.size},
        [this](PoolEntry *entry) {Free(*entry); delete entry;});
  } else {
    return nullptr;
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