#ifndef COLSERVE_RESOURCE_MANAGER_H
#define COLSERVE_RESOURCE_MANAGER_H

#include <server/profiler.h>
#include <server/config.h>

#include <common/cuda_allocator.h>
#include <common/util.h>

#include <memory>
#include <cstddef>

namespace colserve {

class ResourceManager {
 public:
  static void Init() {
    resource_manager_ = std::make_unique<ResourceManager>();
  }
  static double GetFreeMemoryMB(bool verbose);
  static double GetTrainAvailMemoryMB(bool verbose);

  static void InferMemoryChangingLock();
  static void InferMemoryChangingUnlock();
  static bool InferChangeMemoryTryLock();

  static double GetInferMemoryMB();
  static double GetTrainMemoryMB();
  
  ResourceManager();

 private:
  static std::unique_ptr<ResourceManager> resource_manager_;

  std::mutex infer_memory_changing_mut_;
};

}

#endif