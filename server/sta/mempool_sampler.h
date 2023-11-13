#ifndef COLSERVE_MEMPOOL_SAMPLER_H
#define COLSERVE_MEMPOOL_SAMPLER_H

#include <glog/logging.h>
#include <cassert>
#include <ios>
#include <memory>
#include <fstream>
#include "profiler.h"
#include "sta/cuda_allocator.h"
namespace colserve {
namespace sta {


class MempoolSampler {
private:
    bip::managed_shared_memory segment_;
    bip::interprocess_mutex *mutex_;
    CUDAMemPoolImpl::Addr2Entry *addr2entry_;
    CUDAMemPoolImpl::Size2Entry *size2entry_;
    CUDAMemPoolImpl::PoolEntryImpl *empty_;
    CUDAMemPoolImpl::MemPoolConfig config_;
    CUDAMemPoolImpl::StatMap *stat_;

    CUDAMemPoolImpl::PoolEntryImpl* GetEntry(CUDAMemPoolImpl::EntryHandle handle) {
        return reinterpret_cast<CUDAMemPoolImpl::PoolEntryImpl *>(segment_.get_address_from_handle(handle));
    }

    bip::managed_shared_memory::handle_t GetHandle(CUDAMemPoolImpl::PoolEntryImpl *entry) {
        return segment_.get_handle_from_address(entry);
    }

public:

    MempoolSampler(CUDAMemPoolImpl::MemPoolConfig config): config_(config) {
        config_.shared_memory_name = config_.shared_memory_name + "_" + std::getenv("USER");
        segment_ = bip::managed_shared_memory{bip::open_or_create, config_.shared_memory_name.c_str(),
                                        config_.shared_memory_size};
    auto atomic_init = [&] {
        mutex_ = segment_.find_or_construct<bip::interprocess_mutex>("ShareMutex")();
        addr2entry_ = segment_.find_or_construct<CUDAMemPoolImpl::Addr2Entry>("Addr2Entry")(segment_.get_segment_manager());
        size2entry_ = segment_.find_or_construct<CUDAMemPoolImpl::Size2Entry>("Size2Entry")(segment_.get_segment_manager());
        empty_ = segment_.find_or_construct<CUDAMemPoolImpl::PoolEntryImpl>("EmptyPoolEntryImpl")();
        stat_ = segment_.find_or_construct<CUDAMemPoolImpl::StatMap>("StatMap")();
    };
    segment_.atomic_func(atomic_init);
  
    }
    void DumpBlockList(const std::string &filename) {
        std::fstream stream{filename, std::ios_base::out | std::ios_base::trunc};
        assert(stream.is_open());
        std::cout << "---------- dump whole mempool (all) ----------" << std::endl;
        stream << "start,len,allocated,next,prev,mtype" << std::endl;
        bip::scoped_lock locker(*mutex_);
        for(auto && element : *addr2entry_) {
            auto *entry = GetEntry(element.second);
            stream << entry->addr_offset << "," 
                << entry->nbytes << ","
                << entry->allocate << ","
                << GetEntry(entry->next)->addr_offset << "," 
                << GetEntry(entry->prev)->addr_offset << ","
                << static_cast<unsigned>(entry->mtype) << std::endl;
        }
        stream.close();
    }

    void DumpFreeList(const std::string &filename) {
        std::fstream stream{filename, std::ios_base::out | std::ios_base::trunc};
        assert(stream.is_open());
        std::cout << "---------- dump whole mempool (free) ----------" << std::endl;
        stream << "start,len,allocated,next,prev,mtype" << std::endl;
        bip::scoped_lock locker(*mutex_);
        for(auto && element : *size2entry_) {
            auto *entry = GetEntry(element.second);
            stream << entry->addr_offset << "," 
                << entry->nbytes << ","
                << entry->allocate << ","
                << GetEntry(entry->next)->addr_offset << "," 
                << GetEntry(entry->prev)->addr_offset << ","
                << static_cast<unsigned>(entry->mtype) << std::endl;
        }   
        stream.close();
    }
};

static std::shared_ptr<MempoolSampler> mempool_sampler_instance;


static std::shared_ptr<MempoolSampler> GetMempoolSampler() {
    if (mempool_sampler_instance == nullptr) {
        mempool_sampler_instance = std::make_shared<MempoolSampler>(mempool_config_template);
    }
    return mempool_sampler_instance;
}
// namespace bip = boost::interprocess;
inline void DumpMempoolFreeList(std::string filename) {
    GetMempoolSampler()->DumpFreeList(filename);
}

inline void DumpMempoolBlockList(std::string filename) {
    GetMempoolSampler()->DumpBlockList(filename);
}




}
}

#endif