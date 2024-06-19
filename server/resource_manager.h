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
  static double GetFreeMemoryMB(bool verbose);
  static double GetTrainAvailMemoryMB(bool verbose);

  static void InferMemoryChangingLock();
  static void InferMemoryChangingUnlock();
  static bool InferChangeMemoryTryLock();

  static double GetInferMemoryMB();
  static double GetTrainMemoryMB();
  
  static int GetNumGpu();
  static int GetGpuSystemId(int gpu_id);
  static const std::string& GetGpuSystemUuid(int gpu_id);

  ResourceManager();

 private:
  static std::unique_ptr<ResourceManager> resource_manager_;

  std::vector<std::string> system_gpu_uuids_;
  std::unordered_map<int, std::pair<int, std::string>> gpu_id_map_; // visible id -> (system id, uuid)

  std::mutex infer_memory_changing_mut_;
};

}

#endif