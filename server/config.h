#ifndef COLSERVE_CONFIG_H
#define COLSERVE_CONFIG_H

#include <atomic>

namespace colserve {

enum class ServeMode {
  kNormal,        // infer/train contention

  kTaskSwitchL1,  // switch infer/train, drop mini-batch
  kTaskSwitchL2,  // switch infer/train, drop epoch
  kTaskSwitchL3,  // switch infer/train, drop training (i.e. pipeswitch)

  kColocateL1,    // colocate infer/train, drop mini-batch -> adjust batch size -> relaunch
  kColocateL2,    // adjust batch at end of mini-batch
};

class Config {
 public:
  static ServeMode serve_mode;
  
  static std::atomic<bool> running;
  
  static bool use_shared_tensor;

  static double cuda_memory_pool_gb;

  static bool infer_raw_blob_alloc;

  inline static bool IsSwitchMode() {
    return Config::serve_mode == ServeMode::kTaskSwitchL1
        || Config::serve_mode == ServeMode::kTaskSwitchL2
        || Config::serve_mode == ServeMode::kTaskSwitchL3;
  }

  inline static bool IsColocateMode() {
    return Config::serve_mode == ServeMode::kColocateL1
        || Config::serve_mode == ServeMode::kColocateL2;
  }

};

}

#endif