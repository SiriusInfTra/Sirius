#include "tvm_allocator.h"

namespace colserve::sta {

std::unique_ptr<TVMAllocator> TVMAllocator::instance_ = nullptr;


TVMAllocator::TVMAllocator(MemPool &mempool): GenericAllocator(mempool, Belong::kInfer) {}

TVMAllocator &TVMAllocator::Get() {
  
  if (instance_ == nullptr) {
    instance_.reset(new TVMAllocator(MemPool::Get()));
  }
  return *instance_;
}

}

