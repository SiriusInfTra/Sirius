#include "torch_allocator.h" 
#include "mempool.h"
#include <iostream>
#include <ostream>
#include <tuple>
#include <memory>
#include <functional>
#include <glog/logging.h>


namespace colserve::sta {



std::unique_ptr<TorchAllocator> TorchAllocator::instance_ = nullptr;


TorchAllocator::TorchAllocator(MemPool &mempool): GenericAllocator(mempool, Belong::kTrain) {
  LOG(INFO) << "Init TorchAllocator";
}

TorchAllocator &TorchAllocator::Get() {
  
  if (instance_ == nullptr) {
    instance_.reset(new TorchAllocator(MemPool::Get()));
  }
  return *instance_;
}






} // namespace colserve::sta
