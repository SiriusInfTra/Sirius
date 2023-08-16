#ifndef COLSERVE_CONFIG_H
#define COLSERVE_CONFIG_H

namespace colserve {

enum class ServeMode {
  kNormal,        // infer/train contention
  kTaskSwitchL1,  // switch infer/train, drop mini-batch
  kTaskSwitchL2,  // switch infer/train, drop epoch
  kTaskSwitchL3,  // switch infer/train, drop training (i.e. pipeswitch)
};

class Config {
 public:
  static ServeMode serve_mode;
  
};

}

#endif