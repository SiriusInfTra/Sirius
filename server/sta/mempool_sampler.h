#ifndef COLSERVE_MEMPOOL_SAMPLER_H
#define COLSERVE_MEMPOOL_SAMPLER_H

#include <glog/logging.h>
#include <cassert>
#include <ios>
#include <memory>
#include <fstream>
#include "profiler.h"
#include "sta/cuda_allocator.h"
#include "sta/mempool.h"
namespace colserve {
namespace sta {


class MempoolSampler {
private:
  MemPool *impl;   
public:
  MempoolSampler(MemPoolConfig config) {
    impl = new MemPool(config, false, true);
  }

  void DumpBlockList(const std::string &filename) {
    std::fstream stream{filename, std::ios_base::out | std::ios_base::trunc};
    assert(stream.is_open());
    std::cout << "---------- dump whole mempool (all) ----------" << std::endl;
    stream << "start,len,mtype,next,prev,mtype" << std::endl;
    bip::scoped_lock locker(*impl->mutex_);
    for(const auto & element : *impl->mem_entry_list_) {
      auto *entry = GetEntry(impl->segment_, element);
      auto *prev = impl->GetPrevEntry(entry);
      auto *next = impl->GetNextEntry(entry);
      stream << entry->addr_offset << "," 
        << entry->nbytes << ","
        << static_cast<int>(entry->mtype) << ","
        << (prev ? next->addr_offset : -1) << "," 
        << (prev ? prev->addr_offset : -1) << ","
        << static_cast<unsigned>(entry->mtype) << std::endl;
    }
    stream.close();
  }

  void DumpFreeList(const std::string &filename) {
    std::fstream stream{filename, std::ios_base::out | std::ios_base::trunc};
    assert(stream.is_open());
    std::cout << "---------- dump whole mempool (free) ----------" << std::endl;
    stream << "start,len,allocated,next,prev,mtype" << std::endl;
    bip::scoped_lock locker(*impl->mutex_);
    for(const auto & element : *impl->freeblock_policy_->freelist_) {
      auto *entry = GetEntry(impl->segment_, element);
      auto *prev = impl->GetPrevEntry(entry);
      auto *next = impl->GetNextEntry(entry);
      stream << entry->addr_offset << "," 
        << entry->nbytes << ","
        << static_cast<int>(entry->mtype) << ","
        << (prev ? next->addr_offset : -1) << "," 
        << (prev ? prev->addr_offset : -1) << ","
        << static_cast<unsigned>(entry->mtype) << std::endl;
    }
    stream.close();
  }
};

static std::shared_ptr<MempoolSampler> mempool_sampler_instance;


static std::shared_ptr<MempoolSampler> GetMempoolSampler() {
  if (mempool_sampler_instance == nullptr) {
    mempool_sampler_instance = std::make_shared<MempoolSampler>(GetDefaultMemPoolConfig(0));
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