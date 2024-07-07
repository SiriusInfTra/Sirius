#ifndef COLSERVE_CONFIG_H
#define COLSERVE_CONFIG_H

#include <atomic>
#include <iostream>
#include <filesystem>

namespace colserve {

enum class ServeMode {
  kNormal,        // infer/train contention

  kTaskSwitchL0,  // task switch w/ memory pressure
  kTaskSwitchL1,  // switch infer/train, drop mini-batch
  kTaskSwitchL2,  // switch infer/train, drop epoch
  kTaskSwitchL3,  // switch infer/train, drop training (i.e. pipeswitch)

  kColocateL1,    // colocate infer/train, drop mini-batch -> adjust batch size -> relaunch
  kColocateL2,    // adjust batch at end of mini-batch
};

// used for multi-GPU serving
enum class ModelPlacePolicy {
  kRoundRobin,
};


inline std::ostream & operator<<(std::ostream &os, const ServeMode &mode) {
  switch (mode) {
    case ServeMode::kNormal:
      os << "kNormal";
      break;
    case ServeMode::kTaskSwitchL0:
      os << "kTaskSwitchL0";
      break;
    case ServeMode::kTaskSwitchL1:
      os << "kTaskSwitchL0";
      break;
    case ServeMode::kTaskSwitchL2:
      os << "kTaskSwitchL2";
      break;
    case ServeMode::kTaskSwitchL3:
      os << "kTaskSwitchL3";
      break;
    case ServeMode::kColocateL1:
      os << "kColocateL1";
    case ServeMode::kColocateL2:
      os << "kColocateL2";
      break;
    default:
      os << "Unknown(" << static_cast<int>(mode) << ")";
      break;
  }
  return os;
}

struct ColocateConfig {
  bool skip_malloc;
  bool skip_loading;
};

class Config {
 public:
  static ServeMode serve_mode;
  static ModelPlacePolicy model_place_policy;

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
  static bool group_param_dump; // enable at the first time to get mod.group
  static bool group_param_nbytes_with_fragment;
  static size_t group_param_load_threshold;

  static bool ondemand_adjust;

  static double cuda_memory_pool_gb;

  static std::string mempool_freelist_policy;

  static bool infer_raw_blob_alloc;

  static bool capture_train_log;

  static std::string profile_log_path;

  static std::string infer_model_config_path;

  static int train_mps_thread_percent;

  static size_t cold_cache_min_capability_nbytes;
  static size_t cold_cache_max_capability_nbytes;

  static size_t infer_alloc_buffer_nbytes;
  static size_t train_over_adjust_nbytes;
  static size_t max_warm_cache_nbytes;
  static double cold_cache_ratio;

  static bool enable_warm_cache_fallback;

  static double task_switch_delay_ms;

  // to avoid memory fragment cause OOM
  static double train_memory_over_predict_mb;

  static double memory_pressure_mb;

  static bool pipeline_load;

  static bool has_warmup;
  static double infer_model_max_idle_ms;

  static bool dump_adjust_info;

  static bool profiler_acquire_resource_lock ;

  static bool skip_set_mps_thread_percent;
  static bool dynamic_sm_partition;
  static bool estimate_infer_model_tpc;
  static double infer_exec_time_estimate_scale;

  static bool dummy_adjust;

  static bool system_initialized;

  static bool profile_gpu_smact;
  static bool profile_gpu_util;
  static bool profile_sm_partition;

  static constexpr bool log_grpc = false;
  static constexpr bool log_train_init = false;
  static constexpr bool log_warm_cache = false;
  static constexpr bool log_cold_cache = false;

  static constexpr bool log_infer_model_init = false;
  static constexpr bool log_infer_model_reclaim = false;
  static constexpr bool log_infer_time = false;
  static constexpr bool log_infer_pipeline_exec = false;
  static constexpr bool log_infer_load_param = false;

  static constexpr bool log_memory_adjust = false;

  static constexpr bool log_controller = false;

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

#define ADJUST_WITH_FLYING 0

}

#endif