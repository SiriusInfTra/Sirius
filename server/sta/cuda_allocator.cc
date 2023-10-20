#include <iostream>
#include <cuda_runtime_api.h>

#include "cuda_allocator.h"
#include <glog/logging.h>
#include <numeric>

#define CUDA_CALL(func) do { \
  auto error = func; \
  if (error != cudaSuccess) { \
    LOG(FATAL) << cudaGetErrorString(error); \
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
      CUDAMemPoolImpl::MemPoolConfig config{
          .cuda_device = 0,
          .cuda_memory_size = nbytes,
          .shared_memory_name = "gpu_colocation_mempool",
          .shared_memory_size = 1024 * 1024 * 1024, /* 1G */
      };
      impl_ = new CUDAMemPoolImpl{config, master};
    }


    std::shared_ptr<CUDAMemPool::PoolEntry> CUDAMemPool::Alloc(std::size_t nbytes) {
      return impl_->Alloc(nbytes);
    }

    std::shared_ptr<CUDAMemPool::PoolEntry> CUDAMemPool::Resize(
        std::shared_ptr<PoolEntry> entry, std::size_t nbytes) {
      // TODO: handle reallocate
      auto ptr = impl_->Alloc(nbytes);
      impl_->CopyFromTo(ptr->addr, entry->addr, entry->nbytes);
      return ptr;
    }

    void CUDAMemPool::CopyFromTo(std::shared_ptr<PoolEntry> src, std::shared_ptr<PoolEntry> dst) {
      if (src == nullptr || dst == nullptr) return;
      if (src->addr == dst->addr) return;
      impl_->CopyFromTo(dst->addr, src->addr, std::min(src->nbytes, dst->nbytes));
    }


    std::shared_ptr<CUDAMemPool::PoolEntry> CUDAMemPool::RawAlloc(size_t nbytes) {
      void *ptr;
      CUDA_CALL(cudaSetDevice(0));
      CUDA_CALL(cudaMalloc(&ptr, nbytes));
      return std::shared_ptr<PoolEntry>(
          new PoolEntry{ptr, nbytes}, [](PoolEntry *entry) {
            CUDA_CALL(cudaSetDevice(0));
            CUDA_CALL(cudaFree(entry->addr));
            delete entry;
          });
    }


    CUDAMemPoolImpl::PoolEntryImpl *CUDAMemPoolImpl::GetEntry(CUDAMemPoolImpl::EntryHandle handle) {
      return reinterpret_cast<PoolEntryImpl *>(segment_.get_address_from_handle(handle));
    }

    bip::managed_shared_memory::handle_t CUDAMemPoolImpl::GetHandle(CUDAMemPoolImpl::PoolEntryImpl *entry) {
      return segment_.get_handle_from_address(entry);
    }

    void
    CUDAMemPoolImpl::UpdateFreeEntrySize(Size2Entry::iterator iter,
                                         CUDAMemPoolImpl::PoolEntryImpl *entry, size_t newSize) {
      CHECK(iter != size2entry_->cend());
      CHECK_EQ(entry->allocate, false);

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

    void CUDAMemPoolImpl::UpdateEntryAddr(const Addr2Entry::iterator &iter, std::ptrdiff_t newAddr) {
      CHECK(iter != addr2entry_->cend());
      auto entry = GetEntry(iter->second);
      addr2entry_->erase(iter);
      entry->addr_offset = newAddr;
      addr2entry_->insert(std::pair{entry->addr_offset, segment_.get_handle_from_address(entry)});
    }

    void
    CUDAMemPoolImpl::ConnectPoolEntryHandle(CUDAMemPoolImpl::PoolEntryImpl *eh1,
                                            CUDAMemPoolImpl::PoolEntryImpl *eh2) {
      if (eh1 != empty_) {
        eh1->next = GetHandle(eh2);
      }
      if (eh2 != empty_) {
        eh2->prev = GetHandle(eh1);
      }
    }

    void CUDAMemPoolImpl::CheckMemPool() {
      for (auto &&p: *addr2entry_) {
        auto &addr = p.first;
        auto *entry = GetEntry(p.second);
        CHECK_EQ(addr, entry->addr_offset);
        if (auto *prev = GetEntry(entry->prev); prev != empty_) {
          CHECK_LE(prev->addr_offset, addr);
          CHECK_EQ(prev->addr_offset + prev->nbytes, addr);
        }
        if (auto *next = GetEntry(entry->next); next != empty_) {
          CHECK_GE(next->addr_offset, addr);
          CHECK_EQ(next->addr_offset, addr + entry->nbytes);
        }
      }
      for (auto &&p: *size2entry_) {
        auto &nbytes = p.first;
        auto *entry = GetEntry(p.second);
        CHECK_EQ(entry->allocate, false);
        CHECK_EQ(entry->nbytes, nbytes);
      }
    }

    std::shared_ptr<CUDAMemPool::PoolEntry> CUDAMemPoolImpl::MakeSharedPtr(CUDAMemPoolImpl::PoolEntryImpl *eh) {
      auto *entry = new PoolEntry{reinterpret_cast<std::byte *>(devPtr_) + eh->addr_offset, eh->nbytes};
      auto free = [this, eh](PoolEntry *entry) {
        Free(eh);
        delete entry;
      };
      return {entry, free};
    }

    CUDAMemPoolImpl::CUDAMemPoolImpl(CUDAMemPoolImpl::MemPoolConfig config, bool force_master) : config_(
        std::move(config)), devPtr_(nullptr) {
      config_.shared_memory_name = config_.shared_memory_name + "_" + std::getenv("USER");
      if (force_master) {
        bip::shared_memory_object::remove(config_.shared_memory_name.c_str());
      }
      segment_ = bip::managed_shared_memory{bip::open_or_create, config_.shared_memory_name.c_str(),
                                            config_.shared_memory_size};
      auto atomic_init = [&] {
        mutex_ = segment_.find_or_construct<bip::interprocess_mutex>("ShareMutex")();
        addr2entry_ = segment_.find_or_construct<Addr2Entry>("Addr2Entry")(segment_.get_segment_manager());
        size2entry_ = segment_.find_or_construct<Size2Entry>("Size2Entry")(segment_.get_segment_manager());
        ref_count_ = segment_.find_or_construct<RefCount>("RefCount")(0);
        empty_ = segment_.find_or_construct<PoolEntryImpl>("RefCount")();
        cuda_mem_handle_ = segment_.find_or_construct<cudaIpcMemHandle_t>("CudaMemHandle")();
      };
      segment_.atomic_func(atomic_init);
      CUDA_CALL(cudaSetDevice(config.cuda_device));
      CUDA_CALL(cudaStreamCreate(&cuda_memcpy_stream_));
      bip::scoped_lock locker(*mutex_);
      master_ = (*ref_count_)++ == 0;
      if (master_) {
        auto *entry = reinterpret_cast<PoolEntryImpl *>(segment_.allocate(sizeof(PoolEntryImpl)));
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

        LOG(INFO) << "[mempool] init master.";
      } else {
        CUDA_CALL(cudaIpcOpenMemHandle(&devPtr_, *cuda_mem_handle_, cudaIpcMemLazyEnablePeerAccess));
        LOG(INFO) << "[mempool] init slave.";
      }
    }

    CUDAMemPoolImpl::~CUDAMemPoolImpl() {
      CUDA_CALL(cudaStreamDestroy(cuda_memcpy_stream_));
      if (master_) {
        RefCount refCount;
        auto getRefCount = [&] {
          bip::scoped_lock locker(*mutex_);
          return *ref_count_;
        };
        while ((refCount = getRefCount()) > 1) {
          LOG(INFO) << "[mempool] master wait slave shutdown, ref_count = " << refCount << ".";
          std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        CUDA_CALL(cudaFree(devPtr_));
        bip::shared_memory_object::remove(config_.shared_memory_name.c_str());
        LOG(INFO) << "[mempool] free master.";
      } else {
        bip::scoped_lock locker(*mutex_);
        --(*ref_count_);
        CUDA_CALL(cudaIpcCloseMemHandle(devPtr_));
        LOG(INFO) << "[mempool] free slave.";
      }
    }

    std::shared_ptr<CUDAMemPool::PoolEntry> CUDAMemPoolImpl::Alloc(std::size_t nbytes) {
      if (nbytes == 0) { return {nullptr}; }
      bip::scoped_lock locker(*mutex_);
      CheckMemPool();
      nbytes = (nbytes + 1023) / 1024 * 1024; // simple align to 1024B
      auto iter = size2entry_->lower_bound(nbytes);
      if (iter == size2entry_->cend()) {
        iter = size2entry_->lower_bound(nbytes);
      }
      if (iter == size2entry_->cend()) {
        DumpSummary();
        LOG(FATAL) << "request size " << nbytes << " out of free gpu memory";
      }
      auto entry = GetEntry(iter->second);
      if (nbytes == iter->first) {
        CHECK_EQ(entry->allocate, false);
        entry->allocate = true;
        size2entry_->erase(iter);
        return MakeSharedPtr(entry);
      }

      size_t nbytes_rest = entry->nbytes - nbytes;
      entry->nbytes = nbytes;
      entry->allocate = true;
      size2entry_->erase(iter);

      auto *split = reinterpret_cast<PoolEntryImpl *>(segment_.allocate(sizeof(PoolEntryImpl)));
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

    std::shared_ptr<CUDAMemPool::PoolEntry> CUDAMemPoolImpl::Resize(std::shared_ptr<PoolEntry> en, std::size_t nbytes) {
      throw std::runtime_error("not implement.");
//      if (en->nbytes == nbytes) { return en; }
//      auto iter = size2entry_->find(reinterpret_cast<std::byte*>(en->addr) - reinterpret_cast<std::byte*>(devPtr_));
//      CHECK(iter != size2entry_->cend());
//      CheckMemPool();
//      auto *entry = GetEntry(iter->second);
//      if (auto *prev = GetEntry(entry->prev); prev != empty_ && !prev->allocate) { /* merge prev */
//        addr2entry_->erase(entry->addr_offset);
//        UpdateFreeEntrySize(size2entry_->find(prev->nbytes), prev, prev->nbytes + entry->nbytes);
//        ConnectPoolEntryHandle(prev, GetEntry(entry->next));
//        segment_.deallocate(entry);
//      } else if (auto *next = GetEntry(entry->next); next != empty_ && !next->allocate) { /* merge next */
//        addr2entry_->erase(entry->addr_offset);
//        UpdateFreeEntrySize(size2entry_->find(next->nbytes), next, next->nbytes + entry->nbytes);
//        UpdateEntryAddr(addr2entry_->find(next->addr_offset),
//                        next->addr_offset - static_cast<std::ptrdiff_t>(entry->nbytes));
//        ConnectPoolEntryHandle(GetEntry(entry->prev), next);
//        segment_.deallocate(entry);
//      } else { /* add */
//        entry->allocate = false;
//        size2entry_->insert(std::pair{entry->nbytes, GetHandle(entry)});
//      }
    }

    void CUDAMemPoolImpl::Free(CUDAMemPoolImpl::PoolEntryImpl *entry) {
      bip::scoped_lock locker(*mutex_);
      CheckMemPool();
      CHECK_EQ(entry->allocate, true);
      if (auto *prev = GetEntry(entry->prev); prev != empty_ && !prev->allocate) { /* merge prev */
        addr2entry_->erase(entry->addr_offset);
        UpdateFreeEntrySize(size2entry_->find(prev->nbytes), prev, prev->nbytes + entry->nbytes);
        ConnectPoolEntryHandle(prev, GetEntry(entry->next));
        segment_.deallocate(entry);
      } else if (auto *next = GetEntry(entry->next); next != empty_ && !next->allocate) { /* merge next */
        addr2entry_->erase(entry->addr_offset);
        UpdateFreeEntrySize(size2entry_->find(next->nbytes), next, next->nbytes + entry->nbytes);
        UpdateEntryAddr(addr2entry_->find(next->addr_offset),
                        next->addr_offset - static_cast<std::ptrdiff_t>(entry->nbytes));
        ConnectPoolEntryHandle(GetEntry(entry->prev), next);
        segment_.deallocate(entry);
      } else { /* add */
        entry->allocate = false;
        size2entry_->insert(std::pair{entry->nbytes, GetHandle(entry)});
      }
    }

    void CUDAMemPoolImpl::DumpSummary() {
      LOG(INFO) << "---------- mempool summary ----------";
      LOG(INFO) << "free blocks: " << size2entry_->size();
      LOG(INFO) << "free size: " << std::accumulate(size2entry_->cbegin(), size2entry_->cend(), 0L,
        [](auto acc, auto &&pair) { return acc + pair.second; });
      LOG(INFO) << "largest free block size: " << (--size2entry_->cend())->first;
      LOG(INFO) << "total blocks: " << addr2entry_->size();
      LOG(INFO) << "total size: " << std::accumulate(addr2entry_->cbegin(), addr2entry_->cend(), 0L,
                [&](auto acc, auto &&pair) { return acc + GetEntry(pair.second)->nbytes; });
    }

    void CUDAMemPoolImpl::CopyFromTo(void *dst_dev_ptr, void *src_dev_ptr, size_t nbytes) {
      CUDA_CALL(cudaMemcpyAsync(dst_dev_ptr, src_dev_ptr, nbytes, cudaMemcpyDefault, cuda_memcpy_stream_));
      CUDA_CALL(cudaStreamSynchronize(cuda_memcpy_stream_));
    }


  }
}