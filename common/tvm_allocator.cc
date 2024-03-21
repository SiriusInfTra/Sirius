#include "tvm_allocator.h"
#include "mempool.h"
#include <algorithm>
#include <iterator>

namespace colserve::sta {

std::unique_ptr<TVMAllocator> TVMAllocator::instance_ = nullptr;


TVMAllocator::TVMAllocator(MemPool &mempool, bool for_train, bip::scoped_lock<bip::interprocess_mutex> &lock): GenericAllocator(mempool, Belong::kInfer, lock) {
  LOG(INFO) << log_prefix_ << "Init TVMAllocator with args: for_train = " << for_train << ".";
  std::vector<PhyMem *> phy_mem_list;
  for(auto && phymem : mempool.GetPhyMemList()) {
    phy_mem_list.push_back(&phymem);
  }
  ExpandMemorySpace(phy_mem_list, phy_mem_list.size());
  if (entry_list_.GetEntry(0) == nullptr) {
    LOG(INFO) << log_prefix_ << "Init TVMAllocator entires.";
    auto *first_entry = reinterpret_cast<MemEntry *>(MemPool::Get().GetSharedMemory().allocate(sizeof(MemEntry)));
    first_entry->nbytes      = mempool.mempool_nbytes;
    first_entry->addr_offset = 0;
    
    first_entry->is_free = false;
    first_entry->is_small = false;
    first_entry->is_train = false;
    entry_list_.LinkNewEntry(first_entry);
    free_list_large_.PushFreeEntry(first_entry);
  }

}
 

TVMAllocator &TVMAllocator::Get() {
  
  if (instance_ == nullptr) {
    bip::scoped_lock lock{MemPool::Get().GetMutex(), bip::try_to_lock};
    CHECK(lock.owns()) << "dead lock"; 
    instance_.reset(new TVMAllocator(MemPool::Get(), false, lock));
  }
  return *instance_;
}

void TVMAllocator::Init(bool for_train, bip::scoped_lock<bip::interprocess_mutex> &lock) {
  CHECK(instance_ == nullptr);
  CHECK(lock.owns());
  instance_.reset(new TVMAllocator(MemPool::Get(), true, lock));
}
} // namespace colserve::sta
