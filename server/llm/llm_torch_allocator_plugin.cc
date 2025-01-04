#include <server/llm/llm_torch_allocator_plugin.h>
#include <server/config.h>

#include <common/device_manager.h>
#include <common/cuda_allocator.h>
#include <common/util.h>

#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>

#include <glog/logging.h>
#include <utility>
#include <vector>
#include <utility>
#include <vector>

namespace torch {
namespace cuda {

namespace CUDAColAllocator {

// only aim for debug
#define IGNORE_RECORD_STREAM 0

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

void CUDAColAllocator::init(int device_count) {
  using namespace colserve;
  if (initialized_) {
    return;
  }

  // memory pool will be initialized before llm server module
  // if not, mpool will raise error
  auto mpool = colserve::sta::CUDAMemPool::Get(
    colserve::sta::DeviceManager::GetCurrentDevice());

  initialized_ = true;
  LOG(INFO) << "[LLM Serving] pytorch CUDAColAllocator Initialized";
}

c10::DataPtr CUDAColAllocator::allocate(size_t nbytes) const {
  auto current_device = colserve::sta::DeviceManager::GetCurrentDevice();
  auto current_stream = at::cuda::getCurrentCUDAStream().stream();
  auto addr = const_cast<CUDAColAllocator*>(this)->raw_alloc_with_stream(
      nbytes, current_stream);
  return c10::DataPtr{
      addr, addr, raw_deleter(), 
      c10::Device(c10::DeviceType::CUDA, current_device)};
}

void raw_delete_fn(void* ptr) {
  DLOG(INFO) << "CUDAColAllocator raw_deleter " << ptr;
  CUDAColAllocator::Get()->raw_delete(ptr);
}

c10::DeleterFnPtr CUDAColAllocator::raw_deleter() const {
  return &raw_delete_fn;  
}

void* CUDAColAllocator::raw_alloc(size_t nbytes) {
  auto current_stream = at::cuda::getCurrentCUDAStream().stream();
  return raw_alloc_with_stream(nbytes, current_stream);
}

void* CUDAColAllocator::raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) {
  // assume single stream

  // 0. before allocating, we first check if there existing 
  // delayed delete
  ProcessEvents();

  // 1. first create a pool entry
  auto current_device = sta::DeviceManager::GetCurrentDevice();
  auto entry = sta::CUDAMemPool::Get(current_device)->AllocWithStream(
      nbytes, sta::MemType::kInfer, stream, false);
  DLOG(INFO) << "[CUDAColAllocator] device " << current_device
             << " stream " << stream
             << " alloc " << sta::PrintByte(nbytes) 
             << " addr " << entry->addr 
             << " nbytes " << sta::PrintByte(entry->nbytes);
  
  // 2. create extra data structure
  entry->extra_data = new TorchMemBlockExtraData{};

  // 3. record the entry in the allocator
  std::unique_lock<std::mutex> lock(entry_mutex_);
  auto res = entry_map_.insert(std::make_pair(entry->addr, entry)); 
  CHECK(res.second == true || entry->addr == nullptr)
      << " addr " << entry->addr 
      << " nbytes " << sta::PrintByte(entry->nbytes) 
      << " abnormal raw alloc";

  return entry->addr;
}

void CUDAColAllocator::raw_delete(void* ptr) {
  std::unique_lock<std::mutex> lock(entry_mutex_);
  DLOG(INFO) << "[CUDAColAllocator] raw_delete " << ptr;

  auto it = entry_map_.find(ptr);
  CHECK(it != entry_map_.end()) << "not found ptr " << ptr;

  auto extra_data = GetMemBlockExtraData(it->second.get());
  CHECK(extra_data != nullptr);

  // if there are stream use this ptr, we need to delay the delete
  if (!extra_data->stream_set.empty()) {
    DLOG(INFO) << "[CUDAColAllocator] raw_delete " << ptr 
               << " on device " << it->second->block->device_id
               << " its own stream " << it->second->block->stream
               << " stream_set (" << extra_data->stream_set.begin()->stream() << " ...)"
               << " size " << extra_data->stream_set.size();

    auto stream_set = std::move(extra_data->stream_set);
    CHECK(extra_data->stream_set.empty());

    colserve::sta::DeviceGuard device_guard{it->second->block->device_id};

    cudaEvent_t event;
    COL_CUDA_CALL(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    for (auto & stream : stream_set) {
      COL_CUDA_CALL(cudaEventRecord(event, stream.stream()));
      extra_data->event_count++;
      cuda_stream_events_[stream.stream()].emplace_back(
          it->second.get(), event);
    }
  } else {
    // here the ptr can be safely free
    DeleteEntry(it->second.get());
  }
}

void CUDAColAllocator::emptyCache() {
  ProcessEvents();
  for (int i = 0; i < sta::DeviceManager::GetNumVisibleGpu(); i++) {
    colserve::sta::CUDAMemPool::Get(i)->FreeTrainLocals();
  }
}

void CUDAColAllocator::setMemoryFraction(double fraction, int device) {
  LOG(WARNING) << "setMemoryFraction not implemented";
}

void CUDAColAllocator::cacheInfo(int dev_id, size_t* largestBlock) {
  LOG(WARNING) << "cacheInfo not implemented";
}

void* CUDAColAllocator::getBaseAllocation(void* ptr, size_t* size) {
  LOG(WARNING) << "getBaseAllocation not implemented";
  return nullptr;
}

void CUDAColAllocator::recordStream(const c10::DataPtr& ptr, streamType stream) {
  // LOG(WARNING) << "do nothing, due to single stream assumption"
  //              << " ptr " << ptr.get() << " stream " << stream.stream();
  // return ;
#if IGNORE_RECORD_STREAM
  return;
#endif

  if (!ptr.get()) {
    return;
  }

  if (ptr.get_deleter() != &raw_delete_fn) {
    return;
  }

  std::unique_lock lock{entry_mutex_};
  auto it = entry_map_.find(ptr.get());
  if (it == entry_map_.end()) {
    // the ptr may not belong this allocator
    return;
  }

  if (it->second->block->stream == stream.stream()) {
    // keep same as pytorch, ignore the same stream
    return;
  }
  auto extra_data = GetMemBlockExtraData(it->second.get());
  extra_data->stream_set.insert(stream);
}

c10::cuda::CUDACachingAllocator::DeviceStats CUDAColAllocator::getDeviceStats(int device) {
  LOG(WARNING) << "getDeviceStats not implemented";
  return c10::cuda::CUDACachingAllocator::DeviceStats{};
}

void CUDAColAllocator::resetAccumulatedStats(int device) {
  LOG(WARNING) << "resetAccumulatedStats not implemented";
}

void CUDAColAllocator::resetPeakStats(int device) {
  LOG(WARNING) << "resetPeakStats not implemented";
}

c10::cuda::CUDACachingAllocator::SnapshotInfo CUDAColAllocator::snapshot() {
  LOG(WARNING) << "snapshot not implemented";
  return c10::cuda::CUDACachingAllocator::SnapshotInfo{};
}

void CUDAColAllocator::beginAllocateStreamToPool(
    int device,
    cudaStream_t stream,
    c10::cuda::MempoolId_t mempool_id) {
  LOG(WARNING) << "beginAllocateStreamToPool not implemented";
}

void CUDAColAllocator::endAllocateStreamToPool(
    int device, cudaStream_t stream) {
  LOG(WARNING) << "endAllocateStreamToPool not implemented";
}

void CUDAColAllocator::releasePool(int device, c10::cuda::MempoolId_t mempool_id) {
  LOG(WARNING) << "releasePool not implemented";
}

std::shared_ptr<void> CUDAColAllocator::getIpcDevPtr(std::string handle) {
  LOG(WARNING) << "getIpcDevPtr not implemented";
  return nullptr;
}

void CUDAColAllocator::recordHistory(
    bool enabled,
      c10::cuda::CUDACachingAllocator::CreateContextFn context_recorder,
      size_t alloc_trace_max_entries,
      c10::cuda::CUDACachingAllocator::RecordContext when) {
  LOG(WARNING) << "recordHistory not implemented";
};

void CUDAColAllocator::attachOutOfMemoryObserver(
    c10::cuda::CUDACachingAllocator::OutOfMemoryObserver observer) {
  LOG(WARNING) << "attachOutOfMemoryObserver not implemented";
};

void CUDAColAllocator::enablePeerAccess(int dev, int dev_to_access) {
  LOG(WARNING) << "enablePeerAccess not implemented";
}

cudaError_t CUDAColAllocator::memcpyAsync(
    void* dst,
    int dstDevice,
    const void* src,
    int srcDevice,
    size_t count,
    cudaStream_t stream,
    bool p2p_enabled) {
  return cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream);
}

std::shared_ptr<c10::cuda::CUDACachingAllocator::AllocatorState> 
CUDAColAllocator::getCheckpointState(
    int device,
    c10::cuda::MempoolId_t id) {
  LOG(WARNING) << "getCheckpointState not implemented";
  return nullptr;
}

c10::cuda::CUDACachingAllocator::CheckpointDelta 
CUDAColAllocator::setCheckpointPoolState(
    int device,
    std::shared_ptr<c10::cuda::CUDACachingAllocator::AllocatorState> pps) {
  LOG(WARNING) << "setCheckpointPoolState not implemented";
  return {};
}

bool CUDAColAllocator::initialized() {
  return initialized_;
}

std::string CUDAColAllocator::name() {
  return "CUDAColAllocator";
}


TorchMemBlockExtraData* 
CUDAColAllocator::GetMemBlockExtraData(
    colserve::sta::CUDAMemPool::PoolEntry* entry) {
  return static_cast<TorchMemBlockExtraData*>(entry->extra_data);
}

void CUDAColAllocator::ProcessEvents() {
  for (auto it = cuda_stream_events_.begin(); it != cuda_stream_events_.end(); ) {
    while (!it->second.empty()) {
      auto &e = it->second.front();
      auto entry = e.first;
      cudaEvent_t event = e.second;

      cudaError_t err = cudaEventQuery(event);
      if (err == cudaErrorNotReady) {
        cudaGetLastError();
        break;
      } else if (err != cudaSuccess) {
        COL_CUDA_CALL(err);
      }

      auto extra_data = GetMemBlockExtraData(entry);
      extra_data->event_count--;
      if (extra_data->event_count == 0) {
        DeleteEntry(entry);
      }
      it->second.pop_front();
    }

    if (it->second.empty()) {
      it = cuda_stream_events_.erase(it);
    } else {
      ++it;
    }
  }
}

void CUDAColAllocator::DeleteEntry(colserve::sta::CUDAMemPool::PoolEntry* entry) {
  DLOG(INFO) << "CUDAColAllocator DeleteEntry " << entry->addr;
  
  auto extra_data = GetMemBlockExtraData(entry);
  
  // 1. clear attached extra data structure
  delete extra_data;

  // 2. remove the pool entry
  entry_map_.erase(entry->addr);
}

} // namespace CUDAColAllocator
} // namespace cuda
} // namespace torch
