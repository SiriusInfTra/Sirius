#include <common/util.h>

#include "cuda_allocator_plugin.h"
#include "config.h"

#include <glog/logging.h>

namespace torch {
namespace cuda {

namespace CUDAColAllocator {

using namespace colserve;

std::shared_ptr<CUDAColAllocator> CUDAColAllocator::cuda_col_allocator_ = nullptr;

void CUDAColAllocator::SetCurrentAllocator() {
  CHECK(cuda_col_allocator_.get() != nullptr)
    << "CUDAColAllocator is not initialized";
  CHECK(!c10::cuda::CUDACachingAllocator::allocator.load()->initialized())
    << "Can't swap an already initialized allocator";
  c10::cuda::CUDACachingAllocator::allocator.store(cuda_col_allocator_.get());
}

CUDAColAllocator::CUDAColAllocator() {
  LOG(INFO) << "CUDAColAllocator" << std::endl;
  init(1); // for a single gpu
}

void CUDAColAllocator::init(int device_count) {
  if (initialized_) {
    return;
  }
  // init
  LOG(INFO) << "CUDAColAllocator Initialized";  
  // auto has_server_env = std::getenv("SHARED_TENSOR_HAS_SERVER");
  // auto pool_size_env = std::getenv("SHARED_TENSOR_POOL_GB");
  auto pool_freelist_policy_env = std::getenv("SHARED_TENSOR_POOL_FREELIST_POLICY");
  std::string pool_freelist_policy_str = pool_freelist_policy_env ? 
                                          std::string(pool_freelist_policy_env) :
                                          "best-fit";
  auto pool_freelist_policy = colserve::sta::getFreeListPolicy(pool_freelist_policy_str);
  // bool has_server = has_server_env && std::string(has_server_env) == "1"; 
  // double pool_gb = 12; 
  // if (!has_server && !pool_size_env) {
  // LOG(INFO) << "SHARED_TENSOR_POOL_GB not set, use default 12GB"; 
  // } else if (pool_size_env) {
  //   pool_gb = std::stod(pool_size_env);
  // }
  size_t pool_nbytes = static_cast<size_t>(torch_col::shared_tensor_pool_gb * 1_GB); 
  colserve::sta::InitMemoryPool(pool_nbytes, !torch_col::has_shared_tensor_server, 
                                false, pool_freelist_policy);

  initialized_ = true;
}

c10::DataPtr CUDAColAllocator::allocate(size_t nbytes) const {
  auto entry = sta::CUDAMemPool::Get()->Alloc(nbytes, colserve::sta::MemType::kTrain, false);
  
  auto addr = const_cast<CUDAColAllocator*>(this)->raw_alloc(nbytes);
  return c10::DataPtr{addr, addr, raw_deleter(), c10::Device(c10::DeviceType::CUDA, 0)};
}

c10::DeleterFnPtr CUDAColAllocator::raw_deleter() const {
  return [](void* ptr) {
    cuda_col_allocator_->raw_delete(ptr);
  };
}

void* CUDAColAllocator::raw_alloc(size_t nbytes) {
  auto entry = sta::CUDAMemPool::Get()->Alloc(nbytes, colserve::sta::MemType::kTrain, false);
  
  std::unique_lock<std::mutex> lock(entry_mutex_);
  entry_map_[entry->addr] = entry;
  return entry->addr;
}

void* CUDAColAllocator::raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) {
  return raw_alloc(nbytes); // assume single stream
}

void CUDAColAllocator::raw_delete(void* ptr) {
  std::unique_lock<std::mutex> lock(entry_mutex_);
  auto it = entry_map_.find(ptr);
  if (it != entry_map_.end()) {
    entry_map_.erase(it);
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

void CUDAColAllocator::TagIntermMemory(at::Storage storage) {
  std::unique_lock lock(interm_memory_mutex_);
  interm_memories_.push_back(storage);
};

void CUDAColAllocator::ReleaseIntermMemory() {
  std::unique_lock lock(interm_memory_mutex_);
  for (auto &s : interm_memories_) {
    s.unsafeGetStorageImpl()->reset();
  }
}

void CUDAColAllocator::UntagIntermMemory() {
  std::unique_lock lock(interm_memory_mutex_);
  interm_memories_.clear();
}

}
}
}