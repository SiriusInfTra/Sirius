#ifndef TORCH_COL_CONFIG_H
#define TORCH_COL_CONFIG_H

#include <iostream>

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

  static int release_interm_memory_by_grad_fn;
  static int release_interm_memory_by_tagging;
  static int use_fbward_hook;

  static int train_rank;
  static int train_world_size;
  
  static int configured;

  static void InitConfig(int train_rank, int train_world_size);
  static inline int IsConfigured() { return configured; }

  static inline int IsEnableSharedTensor() { return use_shared_tensor; }
  static inline std::string GetHookMode() { return hook_mode; }
  static inline int IsEnableXsched() { return colocate_use_xsched; }
  static inline int IsEnableDynamicSmPartition() { return dynamic_sm_partition; }
  static inline int IsReleaseIntermMemoryByGradFn() { return release_interm_memory_by_grad_fn; }
  static inline void SetReleaseIntermMemoryByGradFn(int value) { 
    release_interm_memory_by_grad_fn = value; 
  }
  static inline int IsReleaseIntermMemoryByTagging() { return release_interm_memory_by_tagging; }
  static inline void SetReleaseIntermMemoryByTagging(int value) { 
    release_interm_memory_by_tagging = value;
  }
  static inline int IsEnableFbwardHook() { return use_fbward_hook; }
  // static inline int GetKillBatchOnRecv() { return kill_batch_on_recv; }

  static inline int GetTrainRank() { return train_rank; }
  static inline void SetTrainRank(int rank) { train_rank = rank; }
  static inline int GetTrainWorldSize() { return train_world_size; }
  static inline void SetTrainWorldSize(int size) { train_world_size = size; }
};

}

#endif