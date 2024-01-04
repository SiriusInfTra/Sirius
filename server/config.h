#ifndef COLSERVE_CONFIG_H
#define COLSERVE_CONFIG_H

#include <atomic>
#include <iostream>
#include <filesystem>

namespace colserve {

#define NVML_CALL(func) do{ \
    auto error = func; \
    if (error != NVML_SUCCESS) { \
      LOG(FATAL) << #func << " " << nvmlErrorString(error); \
      exit(EXIT_FAILURE); \
    } \
  } while(0);

#define CUDA_CALL(func) do { \
    auto error = func; \
    if (error != cudaSuccess) { \
    LOG(FATAL) << #func << " " << cudaGetErrorString(error); \
      exit(EXIT_FAILURE); \
    } \
  } while (0);

enum class ServeMode {
  kNormal,        // infer/train contention

  kTaskSwitchL0,  // task switch w/ memory pressure
  kTaskSwitchL1,  // switch infer/train, drop mini-batch
  kTaskSwitchL2,  // switch infer/train, drop epoch
  kTaskSwitchL3,  // switch infer/train, drop training (i.e. pipeswitch)

  kColocateL1,    // colocate infer/train, drop mini-batch -> adjust batch size -> relaunch
  kColocateL2,    // adjust batch at end of mini-batch
};

struct ColocateConfig {
  bool skip_malloc;
  bool skip_loading;
};

class Config {
 public:
  static ServeMode serve_mode;
  static std::filesystem::path binary_directory;

  static ColocateConfig colocate_config;

  static bool check_mps;
  
  static std::atomic<bool> running;

  static std::string train_profile;

  static bool use_xsched;

  static bool use_shared_tensor;
  static bool use_shared_tensor_infer;
  static bool use_shared_tensor_train;
  static bool better_alloc;
  static size_t better_alloc_threshold;

  static bool group_param_load;
  static size_t group_param_load_threshold;

  static bool ondemand_adjust;

  static double cuda_memory_pool_gb;

  static std::string mempool_freelist_policy;

  static bool infer_raw_blob_alloc;

  static bool capture_train_log;

  static std::string profile_log_path;

  static std::string infer_model_config_path;

  static int train_mps_thread_percent;

  static size_t max_cache_nbytes;

  static double task_switch_delay_ms;


  // to avoid memory fragment cause OOM
  static double train_memory_over_predict_mb;

  static double memory_pressure_mb;

  static bool pipeline_load;

  static bool system_initialized;

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