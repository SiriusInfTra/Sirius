//
// Created by wyk on 10/16/23.
//

#pragma once

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/container/map.hpp>
#include <boost/unordered_map.hpp>
#include <boost/functional/hash.hpp>
#include <boost/thread/lock_guard.hpp>


#include <cuda_runtime_api.h>

#include <thread>
#include <chrono>
#include <iostream>

#define CUDA_CALL(func) do { \
  auto error = func; \
  if (error != cudaSuccess) { \
    std::cout << cudaGetErrorString(error); \
    exit(EXIT_FAILURE); \
  } \
  } while (0)


namespace bip = boost::interprocess;

class CUDAMemPoolImpl {
public:
  struct MemPoolConfig {
    int cuda_device;
    size_t cuda_memory_size;
    std::string shared_memory_name;
    size_t shared_memory_size;
  };
  struct PoolEntry {
    void *addr;
    std::size_t nbytes;
  };

private:
  using Addr2EntryType = std::pair<std::ptrdiff_t, bip::managed_shared_memory::handle_t>;
  using Addr2EntryAllocator = bip::allocator<Addr2EntryType, bip::managed_shared_memory::segment_manager>;
  using Addr2Entry = boost::unordered_map<std::ptrdiff_t, bip::managed_shared_memory::handle_t, boost::hash<std::ptrdiff_t>, std::equal_to<>, Addr2EntryAllocator>;

  using Size2EntryType = std::pair<const size_t, bip::managed_shared_memory::handle_t>;
  using Size2EntryAllocator = bip::allocator<Size2EntryType, bip::managed_shared_memory::segment_manager>;
  using Size2Entry = boost::container::multimap<size_t, bip::managed_shared_memory::handle_t, std::less<>, Size2EntryAllocator>;

  using RefCount = int;
  using EntryHandle = bip::managed_shared_memory::handle_t;

  struct PoolEntryHandle {
    std::ptrdiff_t addr_offset;
    std::size_t nbytes;
    EntryHandle prev;
    EntryHandle next;
    bool allocate;
  };

  bip::managed_shared_memory segment_;
  bip::interprocess_mutex *mutex_;
  Addr2Entry *addr2entry_;
  Size2Entry *size2entry_;
  RefCount *ref_count_;
  MemPoolConfig config_;
  PoolEntryHandle *empty_;
  cudaIpcMemHandle_t *cuda_mem_handle_;

  bool master_;
  void *devPtr_{};

  inline PoolEntryHandle *GetEntry(EntryHandle handle) {
    return reinterpret_cast<PoolEntryHandle *>(segment_.get_address_from_handle(handle));
  }

  inline bip::managed_shared_memory::handle_t GetHandle(PoolEntryHandle *entry) {
    return segment_.get_handle_from_address(entry);
  }

  inline void UpdateFreeEntrySize(Size2Entry::iterator iter, PoolEntryHandle *entry, size_t newSize) {
    assert(iter != size2entry_->cend());
    assert(!entry->allocate);

    while (iter != size2entry_->cend() && iter->first == entry->nbytes) {
      auto handle = GetHandle(entry);
      if (iter->second == handle) {
        size2entry_->erase(iter);
        entry->nbytes = newSize;
        size2entry_->insert(std::pair{entry->nbytes, handle});
        return;
      }
      ++iter;
    }
    throw std::runtime_error("fail to remove entry");

  }

  inline void UpdateEntryAddr(const Addr2Entry::iterator &iter, std::ptrdiff_t newAddr) {
    assert(iter != addr2entry_->cend());
    auto entry = GetEntry(iter->second);
    addr2entry_->erase(iter);
    entry->addr_offset = newAddr;
    addr2entry_->insert(std::pair{entry->addr_offset, segment_.get_handle_from_address(entry)});
  }

  inline void ConnectPoolEntryHandle(PoolEntryHandle *eh1, PoolEntryHandle *eh2) {
    if (eh1 != empty_) {
        eh1->next = GetHandle(eh2);
    }
    if (eh2 != empty_) {
        eh2->prev = GetHandle(eh1);
    }
  }

  inline void CheckMemPool() {
    for (auto &&p: *addr2entry_) {
      auto &addr = p.first;
      auto *entry = GetEntry(p.second);
      assert(addr == entry->addr_offset);
      if (auto *prev = GetEntry(entry->prev); prev != empty_) {
        assert(prev->addr_offset < addr);
        assert(prev->addr_offset + prev->nbytes == addr);
      }
      if (auto *next = GetEntry(entry->next); next != empty_) {
        assert(next->addr_offset > addr);
        assert(next->addr_offset == addr + entry->nbytes);
      }
    }
    for (auto &&p: *size2entry_) {
      auto &nbytes = p.first;
      auto *entry = GetEntry(p.second);
      assert(!entry->allocate);
      assert(entry->nbytes == nbytes);
    }
  }

  std::shared_ptr<PoolEntry> MakeSharedPtr(PoolEntryHandle *eh) {
    auto *entry = new PoolEntry{reinterpret_cast<std::byte *>(devPtr_) + eh->addr_offset, eh->nbytes};
    auto free = [this, eh](PoolEntry *entry) {
      Free(eh);
      delete entry;
    };
    return {entry, free};
  }

public:


  explicit CUDAMemPoolImpl(MemPoolConfig config) : config_(std::move(config)), devPtr_(nullptr) {
    config_.shared_memory_name = config_.shared_memory_name + "_" + std::getenv("USER");
    segment_ = bip::managed_shared_memory{bip::open_or_create, config_.shared_memory_name.c_str(),
                        config_.shared_memory_size};
    auto atomic_init = [&] {
      mutex_ = segment_.find_or_construct<bip::interprocess_mutex>("ShareMutex")();
      addr2entry_ = segment_.find_or_construct<Addr2Entry>("Addr2Entry")(segment_.get_segment_manager());
      size2entry_ = segment_.find_or_construct<Size2Entry>("Size2Entry")(segment_.get_segment_manager());
        ref_count_ = segment_.find_or_construct<RefCount>("RefCount")(0);
      empty_ = segment_.find_or_construct<PoolEntryHandle>("RefCount")();
        cuda_mem_handle_ = segment_.find_or_construct<cudaIpcMemHandle_t>("CudaMemHandle")();
    };
    segment_.atomic_func(atomic_init);
    bip::scoped_lock locker(*mutex_);
    master_ = (*ref_count_)++ == 0;
    if (master_) {
      auto *entry = reinterpret_cast<PoolEntryHandle *>(segment_.allocate(sizeof(PoolEntryHandle)));
      entry->nbytes = config_.cuda_memory_size;
      entry->allocate = false;
      entry->prev = GetHandle(empty_);
      entry->next = GetHandle(empty_);
      CUDA_CALL(cudaSetDevice(config_.cuda_device));
      CUDA_CALL(cudaMalloc(&devPtr_, config_.cuda_memory_size));
      CUDA_CALL(cudaIpcGetMemHandle(cuda_mem_handle_, devPtr_));
      auto handle = GetHandle(entry);
      size2entry_->insert(std::pair{entry->nbytes, handle});
      addr2entry_->insert(std::pair{0, handle});
      std::cout << "[mempool] init master." << std::endl;
    } else {
      CUDA_CALL(cudaIpcOpenMemHandle(&devPtr_, *cuda_mem_handle_, cudaIpcMemLazyEnablePeerAccess));
      std::cout << "[mempool] init slave." << std::endl;
    }
  }

  ~CUDAMemPoolImpl() {
    if (master_) {
      RefCount refCount;
      auto getRefCount = [&] {
        bip::scoped_lock locker(*mutex_);
        return *ref_count_;
      };
      while ((refCount = getRefCount()) > 1) {
        std::cout << "[mempool] master wait slave shutdown, ref_count = " << refCount << "." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
      }
      CUDA_CALL(cudaFree(devPtr_));
      bip::shared_memory_object::remove(config_.shared_memory_name.c_str());
      std::cout << "[mempool] free master." << std::endl;
    } else {
      bip::scoped_lock locker(*mutex_);
      --(*ref_count_);
      CUDA_CALL(cudaIpcCloseMemHandle(devPtr_));
      std::cout << "[mempool] free slave." << std::endl;
    }
  }

  std::shared_ptr<PoolEntry> Alloc(std::size_t nbytes) {
    bip::scoped_lock locker(*mutex_);
    CheckMemPool();
    assert(nbytes > 0);
    nbytes = (nbytes + 1023) / 1024 * 1024; // simple align to 1024B
    auto &&iter = size2entry_->lower_bound(nbytes);
    if (iter == size2entry_->cend()) {
      throw std::bad_alloc();
    }
    auto entry = GetEntry(iter->second);
    if (nbytes == iter->first) {
      assert(entry->allocate == false);
      entry->allocate = true;
      size2entry_->erase(iter);
      return MakeSharedPtr(entry);
    }

    size_t nbytes_rest = entry->nbytes - nbytes;
    entry->nbytes = nbytes;
    entry->allocate = true;
    size2entry_->erase(iter);

    auto *split = reinterpret_cast<PoolEntryHandle *>(segment_.allocate(sizeof(PoolEntryHandle)));
    split->nbytes = nbytes_rest;
    split->allocate = false;
    split->addr_offset = entry->addr_offset + static_cast<std::ptrdiff_t>(nbytes);
      ConnectPoolEntryHandle(split, GetEntry(entry->next));
      ConnectPoolEntryHandle(entry, split);
    auto handle = GetHandle(split);
    addr2entry_->insert(std::pair{split->addr_offset, handle});
    size2entry_->insert(std::pair{split->nbytes, handle});
    return MakeSharedPtr(entry);
  }

private:
  void Free(PoolEntryHandle *entry) {
    bip::scoped_lock locker(*mutex_);
    CheckMemPool();
    assert(entry->allocate);
    if (auto *prev = GetEntry(entry->prev); prev != empty_ && !prev->allocate) { /* merge prev */
      addr2entry_->erase(entry->addr_offset);
        UpdateFreeEntrySize(size2entry_->find(prev->nbytes), prev, prev->nbytes + entry->nbytes);
        ConnectPoolEntryHandle(prev, GetEntry(entry->next));
      segment_.deallocate(entry);
    } else if (auto *next = GetEntry(entry->next); next != empty_ && !next->allocate) { /* merge next */
      addr2entry_->erase(entry->addr_offset);
        UpdateFreeEntrySize(size2entry_->find(next->nbytes), next, next->nbytes + entry->nbytes);
        UpdateEntryAddr(addr2entry_->find(next->addr_offset), next->addr_offset - static_cast<std::ptrdiff_t>(entry->nbytes));
        ConnectPoolEntryHandle(GetEntry(entry->prev), next);
      segment_.deallocate(entry);
    } else { /* add */
      entry->allocate = false;
      size2entry_->insert(std::pair{entry->nbytes, GetHandle(entry)});
    }
  }

};
