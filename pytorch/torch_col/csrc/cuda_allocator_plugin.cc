#include <common/util.h>

#include "common/cuda_allocator.h"
#include "cuda_allocator_plugin.h"
#include "config.h"

#include <glog/logging.h>

namespace torch {
namespace cuda {

namespace CUDAColAllocator {

using namespace colserve;

std::shared_ptr<CUDAColAllocator> CUDAColAllocator::cuda_col_allocator_ = nullptr;

CUDAColAllocator* CUDAColAllocator::Get() {
  if (cuda_col_allocator_ == nullptr) {
    LOG(FATAL) << "CUDAColAllocator is not initialized";
  }
  return cuda_col_allocator_.get();
}

void CUDAColAllocator::SetCurrentAllocator() {
  CHECK(cuda_col_allocator_.get() != nullptr)
    << "CUDAColAllocator is not initialized";
  CHECK(!c10::cuda::CUDACachingAllocator::allocator.load()->initialized())
    << "Can't swap an already initialized allocator";
  c10::cuda::CUDACachingAllocator::allocator.store(cuda_col_allocator_.get());
  LOG(INFO) << "CUDAColAllocator is set as current allocator";
}

// CUDAColAllocator::CUDAColAllocator() {
//   // LOG(INFO) << "CUDAColAllocator" << std::endl;
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
            << " infer memory usage " << sta::detail::ByteDisplay(sta::CUDAMemPool::InferMemUsage());
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
  DLOG(INFO) << "CUDAColAllocator alloc " << sta::detail::ByteDisplay(nbytes) 
            << " : addr " << entry->addr << " nbytes " << sta::detail::ByteDisplay(entry->nbytes);
  
  std::unique_lock<std::mutex> lock(entry_mutex_);
  entry_map_[entry->addr] = entry;
  return entry->addr;
}

void* CUDAColAllocator::raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) {
  return raw_alloc(nbytes); // assume single stream
}

void CUDAColAllocator::raw_delete(void* ptr) {
  std::unique_lock<std::mutex> interm_lock{interm_memory_mutex_};
  std::unique_lock<std::mutex> lock(entry_mutex_);
  if (auto it = entry_map_.find(ptr); it != entry_map_.end()) {
    entry_map_.erase(it);
  }
  if (auto it = interm_memories_.find(ptr); it != interm_memories_.end()) {
    interm_memories_.erase(it);
  }
}

void CUDAColAllocator::emptyCache() {
  sta::CUDAMemPool::FreeTrainLocals();
}

void CUDAColAllocator::setMemoryFraction(double fraction, int device) {
  LOG(INFO) << "setMemoryFraction not implemented";
}

void CUDAColAllocator::cacheInfo(int dev_id, size_t* largestBlock) {
  LOG(INFO) << "cacheInfo not implemented";
}

void* CUDAColAllocator::getBaseAllocation(void* ptr, size_t* size) {
  LOG(INFO) << "getBaseAllocation not implemented";
  return nullptr;
}

void CUDAColAllocator::recordStream(const c10::DataPtr&, streamType stream) {
  LOG(INFO) << "do nothing, due to single stream assumption";
}

c10::cuda::CUDACachingAllocator::DeviceStats CUDAColAllocator::getDeviceStats(int device) {
  LOG(INFO) << "getDeviceStats not implemented";
  return c10::cuda::CUDACachingAllocator::DeviceStats{};
}

void CUDAColAllocator::resetAccumulatedStats(int device) {
  LOG(INFO) << "resetAccumulatedStats not implemented";
}

void CUDAColAllocator::resetPeakStats(int device) {
  LOG(INFO) << "resetPeakStats not implemented";
}

c10::cuda::CUDACachingAllocator::SnapshotInfo CUDAColAllocator::snapshot() {
  LOG(INFO) << "snapshot not implemented";
  return c10::cuda::CUDACachingAllocator::SnapshotInfo{};
}

void CUDAColAllocator::notifyCaptureBegin(
    int device,
    c10::cuda::CaptureId_t graph_id,
    c10::cuda::MempoolId_t mempool_id) {
  LOG(INFO) << "notifyCaptureBegin not implemented";
}

void CUDAColAllocator::notifyCaptureAboutToEnd(
    int device,
    c10::cuda::CaptureId_t graph_id) {
  LOG(INFO) << "notifyCaptureAboutToEnd not implemented";
}

void CUDAColAllocator::notifyCaptureEnded(
    int device, c10::cuda::CaptureId_t graph_id) {
  LOG(INFO) << "notifyCaptureEnded not implemented";
}

void CUDAColAllocator::notifyCaptureDestroy(
    int device, c10::cuda::MempoolId_t mempool_id) {
  LOG(INFO) << "notifyCaptureDestroy not implemented";
}

std::shared_ptr<void> CUDAColAllocator::getIpcDevPtr(std::string handle) {
  LOG(INFO) << "getIpcDevPtr not implemented";
  return nullptr;
}

void CUDAColAllocator::recordHistory(
    bool enabled,
    c10::cuda::CUDACachingAllocator::CreateContextFn context_recorder,
    size_t alloc_trace_max_entries,
    bool alloc_trace_record_context) {
  LOG(INFO) << "recordHistory not implemented";
};

void CUDAColAllocator::attachOutOfMemoryObserver(
    c10::cuda::CUDACachingAllocator::OutOfMemoryObserver observer) {
  LOG(INFO) << "attachOutOfMemoryObserver not implemented";
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

void CUDAColAllocator::TagIntermMemory(void* ptr, size_t nbytes, at::Allocator* allocator) {
  static bool warning_logged = false;
  if (allocator != this) {
    if (!warning_logged) {
      LOG(WARNING) << "allocator " << allocator << " may not be CUDAColAllocator";
      warning_logged = true;
    }
    return;
  }
  std::unique_lock lock{interm_memory_mutex_};
  interm_memories_.emplace(ptr, nbytes);
};

void CUDAColAllocator::ReleaseIntermMemory() {
  std::unique_lock interm_memory_lock{interm_memory_mutex_};
  std::unique_lock entry_lock{entry_mutex_};
  for (auto [ptr, nbytes] : interm_memories_) {
    // s.unsafeGetStorageImpl()->reset();
    auto entry_it = entry_map_.find(ptr);
    if (entry_it != entry_map_.end()) {
      CHECK_GE(entry_it->second->nbytes, nbytes) << " ptr " << entry_it->first;
      entry_map_.erase(entry_it);
    }
  }
}

void CUDAColAllocator::UntagIntermMemory() {
  std::unique_lock lock{interm_memory_mutex_};
  interm_memories_.clear();
}

}
}
}