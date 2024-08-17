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
  static double GetFreeMemoryMB(int device_id, bool verbose);
  static double GetTrainAvailMemoryMB(int device_id, bool verbose);

  static double GetInferMemoryMB(int device_id);
  static double GetTrainMemoryMB(int device_id);
  
  ResourceManager();

 private:
  static std::unique_ptr<ResourceManager> resource_manager_;

};

}

#endif