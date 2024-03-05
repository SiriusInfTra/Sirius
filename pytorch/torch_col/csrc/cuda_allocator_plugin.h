#ifndef TORCH_COL_CUDA_ALLOCATOR_PLUGIN_H
#define TORCH_COL_CUDA_ALLOCATOR_PLUGIN_H


#include <c10/core/Allocator.h>
#include <c10/core/Storage.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <common/init.h>

#include <mutex>
#include <unordered_map>


namespace torch {
namespace cuda {

namespace CUDAColAllocator {

using streamType = c10::cuda::CUDAStream;

class CUDAColAllocator : public c10::cuda::CUDACachingAllocator::CUDAAllocator {
 public:
  static void Init() {
    cuda_col_allocator_ = std::make_shared<CUDAColAllocator>();
  };
  static std::shared_ptr<CUDAColAllocator> Get() { return cuda_col_allocator_; };
  static void SetCurrentAllocator();

  CUDAColAllocator();
  
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
  void notifyCaptureBegin(
      int device,
      c10::cuda::CaptureId_t graph_id,
      c10::cuda::MempoolId_t mempool_id) override;
  void notifyCaptureAboutToEnd(
      int device,
      c10::cuda::CaptureId_t graph_id) override;
  void notifyCaptureEnded(int device, c10::cuda::CaptureId_t graph_id)
      override;
  void notifyCaptureDestroy(int device, c10::cuda::MempoolId_t mempool_id) override;
  std::shared_ptr<void> getIpcDevPtr(std::string handle) override;
  void recordHistory(
      bool enabled,
      c10::cuda::CUDACachingAllocator::CreateContextFn context_recorder,
      size_t alloc_trace_max_entries,
      bool alloc_trace_record_context) override;
  void attachOutOfMemoryObserver(
      c10::cuda::CUDACachingAllocator::OutOfMemoryObserver observer) override;
  bool needsPoolSpecificPeerAccess() override;

  std::string name() override;

  void SetTrainModelAllocating(bool v) { train_allocating_ = v; }
  void TagIntermMemory(at::Storage storage);
  void ReleaseIntermMemory();
  void UntagIntermMemory();


 private:
  static std::shared_ptr<CUDAColAllocator> cuda_col_allocator_;

  bool initialized_ = false;
  bool train_allocating_ = false;
  
  std::mutex entry_mutex_;
  std::unordered_map<void*, std::shared_ptr<colserve::sta::CUDAMemPool::PoolEntry>> entry_map_;

  std::mutex interm_memory_mutex_;
  std::vector<at::Storage> interm_memories_;

};

}

}
}

#endif