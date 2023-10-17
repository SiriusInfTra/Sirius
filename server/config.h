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
  kColocateL2,    // adjust batch at end of epoch
};

class Config {
 public:
  static ServeMode serve_mode;
  
  static std::atomic<bool> running;
  
  static bool use_shared_tensor;
};

}

#endif