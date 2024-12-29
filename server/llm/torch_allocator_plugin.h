#ifndef TORCH_COL_CUDA_ALLOCATOR_PLUGIN_H
#define TORCH_COL_CUDA_ALLOCATOR_PLUGIN_H

#include <common/cuda_allocator.h>

#include <ATen/core/TensorBody.h>
#include <c10/core/Allocator.h>
#include <c10/core/Storage.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <ATen/Tensor.h>
#include <ATen/ATen.h>

#include <mutex>
#include <unordered_map>
#include <thread>
#include <chrono>
#include <unordered_set>
#include <queue>


namespace torch {
namespace cuda {

namespace CUDAColAllocator {

using streamType = c10::cuda::CUDAStream;

struct TorchMemBlockExtraData {
  int event_count{0};
  std::unordered_set<streamType> stream_set;
};

class CUDAColAllocator : public c10::cuda::CUDACachingAllocator::CUDAAllocator {
 public:
  static void Init() {
    cuda_col_allocator_ = std::make_shared<CUDAColAllocator>();
  };
  static CUDAColAllocator* Get();
  static void SetCurrentAllocator();

  CUDAColAllocator() = default;
  
  c10::DataPtr allocate(size_t nbytes) const override;
  c10::DeleterFnPtr raw_deleter() const override;

  void* raw_alloc(size_t nbytes) override;
  void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) override;
  void raw_delete(void* ptr) override;
  void init(int device_count) override;
  bool initialized() override;
  void emptyCache() override;
  void setMemoryFraction(double fraction, int device) override;
  void cacheInfo(int dev_id, size_t* largestBlock) override;
  void* getBaseAllocation(void* ptr, size_t* size) override;
  
  void recordStream(const c10::DataPtr&, streamType stream) override;

  c10::cuda::CUDACachingAllocator::DeviceStats getDeviceStats(
      int device) override;
  void resetAccumulatedStats(int device) override;
  void resetPeakStats(int device) override;
  c10::cuda::CUDACachingAllocator::SnapshotInfo snapshot() override;
  void beginAllocateStreamToPool(
      int device,
      cudaStream_t stream,
      c10::cuda::MempoolId_t mempool_id) override;
  void endAllocateStreamToPool(int device, cudaStream_t stream) override;
  void releasePool(int device, c10::cuda::MempoolId_t mempool_id) override;

  std::shared_ptr<void> getIpcDevPtr(std::string handle) override;

  void recordHistory(
      bool enabled,
      c10::cuda::CUDACachingAllocator::CreateContextFn context_recorder,
      size_t alloc_trace_max_entries,
      c10::cuda::CUDACachingAllocator::RecordContext when) override;
  void attachOutOfMemoryObserver(
      c10::cuda::CUDACachingAllocator::OutOfMemoryObserver observer) override;
  void enablePeerAccess(int dev, int dev_to_access) override;

  virtual cudaError_t memcpyAsync(
      void* dst,
      int dstDevice,
      const void* src,
      int srcDevice,
      size_t count,
      cudaStream_t stream,
      bool p2p_enabled) override;
  virtual std::shared_ptr<c10::cuda::CUDACachingAllocator::AllocatorState> getCheckpointState(
      int device,
      c10::cuda::MempoolId_t id) override;
  virtual c10::cuda::CUDACachingAllocator::CheckpointDelta setCheckpointPoolState(
      int device,
      std::shared_ptr<c10::cuda::CUDACachingAllocator::AllocatorState> pps) override;

  std::string name() override;

  void ProcessEvents();
  void DeleteEntry(colserve::sta::CUDAMemPool::PoolEntry* entry);

  TorchMemBlockExtraData* 
  GetMemBlockExtraData(colserve::sta::CUDAMemPool::PoolEntry* entry);


 private:
  static std::shared_ptr<CUDAColAllocator> cuda_col_allocator_;

  bool initialized_ = false;
  
  // first lock interm_memory mutex to avoid dead lock
  std::mutex entry_mutex_;

  std::unordered_map<
      void*, std::shared_ptr<colserve::sta::CUDAMemPool::PoolEntry>
  > entry_map_;
  
  std::unordered_map<void*, std::deque<
    std::pair<colserve::sta::CUDAMemPool::PoolEntry *, cudaEvent_t>
  >> cuda_stream_events_;
};

}

}
}

#endif