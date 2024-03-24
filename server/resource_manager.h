#ifndef COLSERVE_RESOURCE_MANAGER_H
#define COLSERVE_RESOURCE_MANAGER_H

#include <common/cuda_allocator.h>
#include <common/util.h>
#include <server/profiler.h>
#include <server/config.h>

#include <memory>
#include <cstddef>

namespace colserve {

class ResourceManager {
 public:
  static void Init() {
    resource_manager_ = std::make_unique<ResourceManager>();
  }
  static double GetFreeMemoryMB();
  static double GetTrainAvailMemoryMB();

  static void InferMemoryChangingLock() {
    resource_manager_->infer_memory_changing_mut_.lock();
  }
  static void InferMemoryChangingUnlock() {
    resource_manager_->infer_memory_changing_mut_.unlock();
  }
  static bool InferChangeMemoryTryLock() {
    return resource_manager_->infer_memory_changing_mut_.try_lock();
  }

  static double GetInferMemoryMB() {
    using namespace sta;
    if (Config::use_shared_tensor_infer) {
      return ByteToMB(CUDAMemPool::InferMemUsage());
    } else {
      return ByteToMB(Profiler::GetLastInferMem());
    }
  }
  static double GetTrainMemoryMB() {
    using namespace sta;
    if (Config::use_shared_tensor_train) {
      return ByteToMB(CUDAMemPool::TrainAllMemUsage());
    } else {
      return ByteToMB(Profiler::GetLastTrainMem());
    }
  }

 private:
  static std::unique_ptr<ResourceManager> resource_manager_;

  std::mutex infer_memory_changing_mut_;
};

}

#endif