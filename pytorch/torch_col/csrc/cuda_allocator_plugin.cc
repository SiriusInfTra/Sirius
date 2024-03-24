#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>
#include <common/util.h>

#include "common/cuda_allocator.h"
#include "cuda_allocator_plugin.h"
#include "config.h"

#include <glog/logging.h>
#include <utility>
#include <vector>
#include <utility>
#include <vector>

namespace torch {
namespace cuda {

namespace CUDAColAllocator {

using namespace colserve;

std::shared_ptr<CUDAColAllocator> CUDAColAllocator::cuda_col_allocator_ = nullptr;

CUDAColAllocator* CUDAColAllocator::Get() {
  if (cuda_col_allocator_ == nullptr) {
    DLOG(FATAL) << "CUDAColAllocator is not initialized";
  }
  return cuda_col_allocator_.get();
}

void CUDAColAllocator::SetCurrentAllocator() {
  CHECK(cuda_col_allocator_.get() != nullptr)
    << "CUDAColAllocator is not initialized";
  CHECK(!c10::cuda::CUDACachingAllocator::allocator.load()->initialized())
    << "Can't swap an already initialized allocator";
  c10::cuda::CUDACachingAllocator::allocator.store(cuda_col_allocator_.get());
  DLOG(INFO) << "CUDAColAllocator is set as current allocator";
}

// CUDAColAllocator::CUDAColAllocator() {
//   // DLOG(INFO) << "CUDAColAllocator" << std::endl;
//   // init(1); // for a single gpu
// }

void CUDAColAllocator::init(int device_count) {
  using namespace colserve;
  if (initialized_) {
    return;
  }
  // init
  auto pool_freelist_policy_env = std::getenv("SHARED_TENSOR_POOL_FREELIST_POLICY");
  std::string pool_freelist_policy_str = pool_freelist_policy_env ? 
                                          std::string(pool_freelist_policy_env) :
                                          "best-fit";
  // auto pool_freelist_policy = colserve::sta::getFreeListPolicy(pool_freelist_policy_str);
  sta::FreeListPolicyType policy;
  size_t pool_nbytes = static_cast<size_t>(torch_col::shared_tensor_pool_gb * 1_GB); 
  colserve::sta::InitMemoryPool(pool_nbytes, !torch_col::has_shared_tensor_server, 
                                false, policy);

  initialized_ = true;
  LOG(INFO) << "pytorch CUDAColAllocator Initialized, "
            << " infer memory usage " << sta::ByteDisplay(sta::CUDAMemPool::InferMemUsage());
}

c10::DataPtr CUDAColAllocator::allocate(size_t nbytes) const {
  auto addr = const_cast<CUDAColAllocator*>(this)->raw_alloc(nbytes);
  return c10::DataPtr{addr, addr, raw_deleter(), c10::Device(c10::DeviceType::CUDA, 0)};
}

c10::DeleterFnPtr CUDAColAllocator::raw_deleter() const {
  return [](void* ptr) {
    DLOG(INFO) << "CUDAColAllocator raw_deleter " << ptr;
    cuda_col_allocator_->raw_delete(ptr);
  };
}

void* CUDAColAllocator::raw_alloc(size_t nbytes) {
  auto entry = sta::CUDAMemPool::Get()->Alloc(nbytes, colserve::sta::MemType::kTrain, false);
  DLOG(INFO) << "CUDAColAllocator alloc " << sta::ByteDisplay(nbytes) 
            << " : addr " << entry->addr << " nbytes " << sta::ByteDisplay(entry->nbytes);
  
  std::unique_lock<std::mutex> lock(entry_mutex_);
  auto res = entry_map_.insert(std::make_pair(entry->addr, entry)); 
  CHECK(res.second == true || entry->addr == nullptr) 
      << " addr " << entry->addr 
      << " nbytes " << sta::ByteDisplay(entry->nbytes) 
      << " abnormal raw alloc";
  if (train_model_allocating_) {
    train_model_params_.insert(std::make_pair(entry->addr, entry));
  }
  return entry->addr;
}

void* CUDAColAllocator::raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) {
  return raw_alloc(nbytes); // assume single stream
}

void CUDAColAllocator::raw_delete(void* ptr) {
  std::unique_lock<std::mutex> interm_lock{interm_memory_mutex_};
  std::unique_lock<std::mutex> lock(entry_mutex_);
  DLOG(INFO) << "CUDAColAllocator raw_delete " << ptr;
  entry_map_.erase(ptr);
  interm_memories_.erase(ptr);
  train_model_params_.erase(ptr);
  DLOG(INFO) << "CUDAColAllocator raw_delete " << ptr;
  entry_map_.erase(ptr);
  interm_memories_.erase(ptr);
  train_model_params_.erase(ptr);
}

void CUDAColAllocator::emptyCache() {
  sta::CUDAMemPool::FreeTrainLocals();
}

void CUDAColAllocator::setMemoryFraction(double fraction, int device) {
  DLOG(INFO) << "setMemoryFraction not implemented";
}

void CUDAColAllocator::cacheInfo(int dev_id, size_t* largestBlock) {
  DLOG(INFO) << "cacheInfo not implemented";
}

void* CUDAColAllocator::getBaseAllocation(void* ptr, size_t* size) {
  DLOG(INFO) << "getBaseAllocation not implemented";
  return nullptr;
}

void CUDAColAllocator::recordStream(const c10::DataPtr&, streamType stream) {
  DLOG(INFO) << "do nothing, due to single stream assumption";
}

c10::cuda::CUDACachingAllocator::DeviceStats CUDAColAllocator::getDeviceStats(int device) {
  DLOG(INFO) << "getDeviceStats not implemented";
  return c10::cuda::CUDACachingAllocator::DeviceStats{};
}

void CUDAColAllocator::resetAccumulatedStats(int device) {
  DLOG(INFO) << "resetAccumulatedStats not implemented";
}

void CUDAColAllocator::resetPeakStats(int device) {
  DLOG(INFO) << "resetPeakStats not implemented";
}

c10::cuda::CUDACachingAllocator::SnapshotInfo CUDAColAllocator::snapshot() {
  DLOG(INFO) << "snapshot not implemented";
  return c10::cuda::CUDACachingAllocator::SnapshotInfo{};
}

void CUDAColAllocator::notifyCaptureBegin(
    int device,
    c10::cuda::CaptureId_t graph_id,
    c10::cuda::MempoolId_t mempool_id) {
  DLOG(INFO) << "notifyCaptureBegin not implemented";
}

void CUDAColAllocator::notifyCaptureAboutToEnd(
    int device,
    c10::cuda::CaptureId_t graph_id) {
  DLOG(INFO) << "notifyCaptureAboutToEnd not implemented";
}

void CUDAColAllocator::notifyCaptureEnded(
    int device, c10::cuda::CaptureId_t graph_id) {
  DLOG(INFO) << "notifyCaptureEnded not implemented";
}

void CUDAColAllocator::notifyCaptureDestroy(
    int device, c10::cuda::MempoolId_t mempool_id) {
  DLOG(INFO) << "notifyCaptureDestroy not implemented";
}

std::shared_ptr<void> CUDAColAllocator::getIpcDevPtr(std::string handle) {
  DLOG(INFO) << "getIpcDevPtr not implemented";
  return nullptr;
}

void CUDAColAllocator::recordHistory(
    bool enabled,
    c10::cuda::CUDACachingAllocator::CreateContextFn context_recorder,
    size_t alloc_trace_max_entries,
    bool alloc_trace_record_context) {
  DLOG(INFO) << "recordHistory not implemented";
};

void CUDAColAllocator::attachOutOfMemoryObserver(
    c10::cuda::CUDACachingAllocator::OutOfMemoryObserver observer) {
  DLOG(INFO) << "attachOutOfMemoryObserver not implemented";
};

bool CUDAColAllocator::needsPoolSpecificPeerAccess() {
  return false;
}

bool CUDAColAllocator::initialized() {
  return initialized_;
}

std::string CUDAColAllocator::name() {
  return "CUDAColAllocator";
}

void CUDAColAllocator::TagIntermMemory(at::Tensor tensor) {
  static bool warning_DLOGged = false;
  auto & storage = tensor.storage();
  if (storage.allocator() != this) {
    if (!warning_DLOGged) {
      LOG(WARNING) << "allocator " << storage.allocator() << " may not be CUDAColAllocator";
      warning_DLOGged = true;
    }
    return;
  }
  
  std::unique_lock lock{interm_memory_mutex_};
  DLOG(INFO) << "TagIntermMemory emplace " << storage.data() << " nbytes " << storage.nbytes();
  // if (train_model_params_.find(storage.data()) != train_model_params_.cend()) {
  if (train_model_params_.count(storage.data())) {
    DLOG(INFO) << "TagIntermMemory but is train";
  } else {
    interm_memories_[storage.data()].emplace_back(tensor.getIntrusivePtr());
  }
};

void CUDAColAllocator::ReleaseIntermMemory() {
  DLOG(INFO) << "ReleaseIntermMemory";
  std::unique_lock interm_memory_lock{interm_memory_mutex_};
  std::unique_lock entry_lock{entry_mutex_};
  std::vector<at::Storage> plan_release_storage;
  for (auto &&[ptr, weak_ptr_arr] : interm_memories_) {
    for (auto &&weak_ptr : weak_ptr_arr) {
      if (auto tensor_ptr = weak_ptr.lock(); tensor_ptr != nullptr) {
        // since we hold lock, we should not release storage inplace
        plan_release_storage.emplace_back(tensor_ptr->storage());
        DLOG(INFO) << "ReleaseIntermMemory " << ptr << "  release success, nbytes = " << tensor_ptr->storage().nbytes();
        break;
      } else {
        DLOG(INFO) << "ReleaseIntermMemory " << ptr << " already release.";
      }
    }
  }

  interm_memories_.clear();
  interm_memory_lock.unlock();
  entry_lock.unlock();
  for(auto &&storage : plan_release_storage) {
    storage.unsafeGetStorageImpl()->release_resources();
  }

}

void CUDAColAllocator::UntagIntermMemory() {
  std::unique_lock lock{interm_memory_mutex_};
  DLOG(INFO) << "UntagIntermMemory";
  interm_memories_.clear();
}

}
}
}