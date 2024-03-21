#include "torch_allocator.h" 
#include "mempool.h"
#include "tvm_allocator.h"
#include <iostream>
#include <ostream>
#include <tuple>
#include <memory>
#include <functional>
#include <glog/logging.h>


namespace colserve::sta {



std::unique_ptr<TorchAllocator> TorchAllocator::instance_ = nullptr;

TorchAllocator &TorchAllocator::Get() {
  if (instance_ == nullptr) {
    bip::scoped_lock lock{MemPool::Get().GetMutex(), bip::try_to_lock};
    CHECK(lock.owns()) << "dead lock"; 
    instance_.reset(new TorchAllocator(MemPool::Get(), lock));
  }
  return *instance_;
}

TorchAllocator::TorchAllocator(MemPool &mempool, bip::scoped_lock<bip::interprocess_mutex> &lock): GenericAllocator(mempool, Belong::kTrain, lock) {
  LOG(INFO) << log_prefix_ << "Init TorchAllocator.";
  TVMAllocator::Init(true, lock);
}







} // namespace colserve::sta
