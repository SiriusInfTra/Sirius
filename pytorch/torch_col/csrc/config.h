#ifndef TORCH_COL_CONFIG_H
#define TORCH_COL_CONFIG_H

namespace torch_col {

class TorchColConfig {
 public:
  static int use_shared_tensor;
  static double shared_tensor_pool_gb;

  static int has_colocated_infer_server;
  static int has_shared_tensor_server;

  static int colocate_use_xsched;
  static int kill_batch_on_recv;

  static int dynamic_sm_partition;

  static std::string hook_mode;

  static void InitConfig(int use_shared_tensor_);

  static inline std::string GetHookMode() { return hook_mode; }
  static inline int EnableXsched() { return colocate_use_xsched; }
  static inline int EnableDynamicSmPartition() { return dynamic_sm_partition; }
  // static inline int GetKillBatchOnRecv() { return kill_batch_on_recv; }
};

// extern int colocate_use_xsched;
  
// extern int kill_batch_on_recv;

// extern int has_colocated_infer_server;

// extern int has_shared_tensor_server;

// extern double shared_tensor_pool_gb;

// extern bool dynamic_sm_partition;

// void ConfigTorchCol(int use_shared_tensor);

}

#endif