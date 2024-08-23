#include <server/logging_as_glog.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/logging.h>
#include <iostream>
#include <filesystem>
#include <csignal>
#include <grpcpp/grpcpp.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/health_check_service_interface.h>
#include <glog/logging.h>
#include <CLI/CLI.hpp>

#include <common/cuda_allocator.h>
#include <common/sm_partition.h>
#include <common/device_manager.h>
#include <common/util.h>

#include <server/grpc/grpc_server.h>
#include <server/model_store/infer_model_store.h>
#include <server/train_launcher.h>
#include <server/train_adjuster.h>
#include <server/resource_manager.h>
#include <server/control/controller.h>
#include <server/profiler.h>
#include <server/config.h>

#define STREAM_OUTPUT(field) std::cerr << "cfg::" #field "=" << cfg::field << std::endl

CLI::App app{"ColServe"};
std::string mode = "normal";
std::string port = "8080";
int max_live_minute = 20;

void* memory_pressure_ptr = nullptr;

void init_cli_options() {
  app.add_option("-m,--mode", mode,
      "server mode, see detail in server/config.h, default is normal")
      ->check(CLI::IsMember({"normal", 
                             "task-switch-l1", 
                             "task-switch-l2", 
                             "task-switch-l3",
                             "colocate-l1",
                             "colocate-l2"}));
  app.add_option("--mps", colserve::Config::check_mps, 
      "check mps, default is 1");
  app.add_option("--use-sta", 
      colserve::Config::use_shared_tensor, 
      "use shared tensor allocator, default is 1");
  app.add_option("--use-sta-infer", 
      colserve::Config::use_shared_tensor_infer, 
      "use shared tensor allocator in infer, default is 1");
  app.add_option("--use-sta-train", 
      colserve::Config::use_shared_tensor_train,
      "use shared tensor allocator in train, default is 1");
  app.add_option("--cuda-memory-pool-gb", 
      colserve::Config::cuda_memory_pool_gb,
      "cuda memory pool size in GB, default is 12");
  app.add_option("--memory-pool-policy", 
      colserve::Config::mempool_freelist_policy, 
      "cuda memory pool freelist policy, default is best-fit.")
        ->check(CLI::IsMember({"first-fit", "next-fit", "best-fit"}));
  app.add_option("-p,--port", port,
      "gRPC server port, default is 8080");
  app.add_option("--max-live-minute", max_live_minute,
      "max server live minute, default is " 
      + std::to_string(max_live_minute) + " minutes");
  app.add_option("--infer-model-config", 
      colserve::Config::infer_model_config_path, 
      "infer model config path, default is config in models store");
  app.add_option("--profile-log", 
      colserve::Config::profile_log_path, 
      "profile log path, default is server-profile");
  app.add_option("--profile-gpu-smact", 
      colserve::Config::profile_gpu_smact, 
      "profile gpu smact, default is " 
      + std::to_string(colserve::Config::profile_gpu_smact));
  app.add_option("--profile-gpu-util", 
      colserve::Config::profile_gpu_util, 
      "profile gpu util, default is "
      + std::to_string(colserve::Config::profile_gpu_util));
  app.add_option("--profile-sm-partition", 
      colserve::Config::profile_sm_partition, 
      "profile sm partition, default is " 
      + std::to_string(colserve::Config::profile_sm_partition));
  app.add_option("--capture-train-log", 
      colserve::Config::capture_train_log, 
      "capture train log, default is 1");
  app.add_flag("--infer-blob-alloc", 
      colserve::Config::infer_raw_blob_alloc, 
      "infer raw blob alloc, default is false");
  app.add_option("--train-mps-thread-percent", 
      colserve::Config::train_mps_thread_percent, 
      "train mps thread percent, default is None");
  app.add_flag("--colocate-skip-malloc", 
      colserve::Config::colocate_config.skip_malloc, 
      "colocate skip malloc, default is false");
  app.add_flag("--colocate-skip-loading", 
      colserve::Config::colocate_config.skip_loading, 
      "colocate skip loading, default is false");
  app.add_option("--use-xsched", colserve::Config::use_xsched, 
      "use xsched, default is false");
  app.add_option("--dynamic-sm-partition", 
      colserve::Config::dynamic_sm_partition, 
      "dynamic sm partition, default is false");
  app.add_option("--train-profile", colserve::Config::train_profile, 
      "train timeline path, default is train-timeline");
  app.add_option("--max-warm-cache-nbytes", 
      colserve::Config::max_warm_cache_nbytes, 
      "max warm cache nbytes, default is 0*1024*1024*1024(0G).");
  app.add_option("--cold-cache-min-capability-nbytes", 
      colserve::Config::cold_cache_min_capability_nbytes, 
      "min cold cache capability nbytes, default is 0*1024*1024*1024(0G).");
  app.add_option("--cold-cache-max-capability-nbytes", 
      colserve::Config::cold_cache_max_capability_nbytes, 
      "max cold cache capability nbytes, default is 0*1024*1024*1024(0G).");
  app.add_option("--cold-cache-ratio", 
      colserve::Config::cold_cache_ratio, 
      "cold cache ratio, default is 0.3(30%).");
  app.add_option("--memory-pressure-mb", 
      colserve::Config::memory_pressure_mb,
      "memory pressure in MB, default is 0");
  app.add_option("--ondemand-adjust", colserve::Config::ondemand_adjust,
      "ondemand adjust batch size, default is 1");
  app.add_option("--pipeline-load", colserve::Config::pipeline_load,
      "pipeline load, default is 1");
  app.add_option("--train-memory-over-predict-mb", 
      colserve::Config::train_memory_over_predict_mb,
      "train memory over predict in MB, default is 2560");
  app.add_option("--infer-model-max-idle-ms", 
      colserve::Config::infer_model_max_idle_ms,
      "infer model max idle in ms, default is 3000");
  app.add_option("--has-warmup", colserve::Config::has_warmup,
      "has warmup, default is 0");
  app.add_flag("--dump-adjust-info", 
      colserve::Config::dump_adjust_info,
      "dump adjust info, default is 0");
  app.add_flag("--profiler-acquire-resource-lock", 
      colserve::Config::profiler_acquire_resource_lock,
      "profiler acquire resource lock during profiling, "
      "not use for performance eval");
  app.add_flag("--dummy-adjust", colserve::Config::dummy_adjust,
      "dummy adjust for eval, default is 0");
  app.add_flag("--skip-set-mps-thread-percent", 
      colserve::Config::skip_set_mps_thread_percent,
      "skip set mps thread percent, default is 0");

  app.add_flag("--enable-warm-cache-fallback", 
      colserve::Config::enable_warm_cache_fallback,
      "enable warm cache fallback, default is 1");
}

void init_config() {
  using cfg = colserve::Config;
  if (mode == "normal") {
    cfg::serve_mode = colserve::ServeMode::kNormal;
  } else if (mode == "task-switch-l1") {
    cfg::serve_mode = colserve::ServeMode::kTaskSwitchL1;
  } else if (mode == "task-switch-l2") {
    cfg::serve_mode = colserve::ServeMode::kTaskSwitchL2;
  } else if (mode == "task-switch-l3") {
    cfg::serve_mode = colserve::ServeMode::kTaskSwitchL3;
  } else if (mode == "colocate-l1") {
    cfg::serve_mode = colserve::ServeMode::kColocateL1;
  } else if (mode == "colocate-l2") {
    cfg::serve_mode = colserve::ServeMode::kColocateL2;
  } else {
    LOG(FATAL) << "unknown serve mode: " << mode;
  }
  
  colserve::Config::use_shared_tensor_infer = 
      colserve::Config::use_shared_tensor_infer 
      && colserve::Config::use_shared_tensor;

  colserve::Config::use_shared_tensor_train =
      colserve::Config::use_shared_tensor_train 
      && colserve::Config::use_shared_tensor;

  if (!cfg::IsColocateMode()) {
    cfg::ondemand_adjust = false;
  }

  if (!cfg::IsColocateMode()) {
    if (cfg::colocate_config.skip_malloc || cfg::colocate_config.skip_loading) {
      LOG(WARNING) << "ignore colocate skip malloc and loading in non-colocating mode";
    }
    cfg::colocate_config.skip_malloc = false;
    cfg::colocate_config.skip_loading = false;
  }
  if (cfg::colocate_config.skip_loading && !cfg::colocate_config.skip_malloc) {
    LOG(FATAL) << "skip loading must be used with skip malloc";
  } 

  if (cfg::use_shared_tensor) {
    if (cfg::group_param_load && !cfg::better_alloc) {
      LOG(FATAL) << "group param load must be used with better alloc";
    }
  } else {
    cfg::group_param_load = false;
  }

  if ((cfg::IsColocateMode() || cfg::IsSwitchMode()) && 
      cfg::use_shared_tensor &&
      cfg::enable_warm_cache_fallback) {
    CHECK_EQ(cfg::max_warm_cache_nbytes, 0);
    cfg::max_warm_cache_nbytes = 
      static_cast<size_t>((cfg::cuda_memory_pool_gb * 1024 
                           - cfg::train_memory_over_predict_mb) * 1_MB
                           - cfg::cold_cache_max_capability_nbytes); // for warmup
    LOG(INFO) << "enable enable_warm_cache_fallback, "
              << "cache nbytes (used in warmup, conservative estimated) " 
              << colserve::sta::PrintByte(cfg::max_warm_cache_nbytes);
  }

  auto *cuda_device_env = getenv("CUDA_VISIBLE_DEVICES");
  auto *cuda_mps_thread_pct_env = getenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE");
  if (cuda_device_env != nullptr) {
    LOG(INFO) << "ENV::CUDA_VISIBLE_DEVICES: " << cuda_device_env;
  }
  if (cuda_mps_thread_pct_env != nullptr) {
    LOG(INFO) << "ENV::CUDA_MPS_ACTIVE_THREAD_PERCENTAGE: " 
              << cuda_mps_thread_pct_env;
  }

  if (cuda_mps_thread_pct_env == nullptr 
      && cfg::train_mps_thread_percent == -1) {
    LOG(INFO) << "no CUDA_MPS_ACTIVE_THREAD_PERCENTAGE set, "
                 "skip set mps thread percent";
    cfg::skip_set_mps_thread_percent = true;
  }

  if (!cfg::skip_set_mps_thread_percent && cfg::dynamic_sm_partition) {
    LOG(FATAL) << "Dynamic partition SM may not work "
                  "correctly with control of MPS thread percent";
  }
  if (cfg::dynamic_sm_partition && !cfg::use_xsched) {
    LOG(FATAL) << "Dynamic partition SM must work with xsched";
  }

  // log config
  auto col_log_all_env = getenv("COLSERVE_LOG_ALL");
  if (col_log_all_env != nullptr) {
    cfg::log_all = std::string(col_log_all_env) == "1";
  }
  auto col_log_adjust_evn = getenv("COLSERVE_LOG_MEMORY_ADJUST");
  if (col_log_adjust_evn != nullptr) {
    cfg::log_memory_adjust = std::string(col_log_adjust_evn) == "1";
  }
  if (cfg::log_all) {
    cfg::log_all = true;
    cfg::log_grpc = true;
    cfg::log_train_init = true;
    cfg::log_warm_cache = true;
    cfg::log_cold_cache = true;
    cfg::log_infer_model_init = true;
    cfg::log_infer_model_reclaim = true;
    cfg::log_infer_time = true;
    cfg::log_infer_pipeline_exec = true;
    cfg::log_infer_load_param = true;
    cfg::log_memory_adjust = true;
    cfg::log_controller = true;
  }

  std::string header = str(boost::format("%s COLSERVE CONFIG [PID %d] %s") 
                           % std::string(16, '=')
                           % getpid()
                           % std::string(16, '='));
  std::cerr << header << std::endl;
  STREAM_OUTPUT(serve_mode);
  STREAM_OUTPUT(use_shared_tensor);
  STREAM_OUTPUT(use_shared_tensor_infer);
  STREAM_OUTPUT(use_shared_tensor_train);
  STREAM_OUTPUT(use_xsched);
  STREAM_OUTPUT(cuda_memory_pool_gb);
  STREAM_OUTPUT(ondemand_adjust);
  STREAM_OUTPUT(better_alloc);
  STREAM_OUTPUT(group_param_load);
  STREAM_OUTPUT(pipeline_load);
  STREAM_OUTPUT(has_warmup);
  STREAM_OUTPUT(enable_warm_cache_fallback);
  STREAM_OUTPUT(max_warm_cache_nbytes);
  STREAM_OUTPUT(cold_cache_max_capability_nbytes);
  STREAM_OUTPUT(cold_cache_min_capability_nbytes);
  STREAM_OUTPUT(train_over_adjust_nbytes);
  STREAM_OUTPUT(cold_cache_ratio);
  STREAM_OUTPUT(infer_model_max_idle_ms);
  STREAM_OUTPUT(dynamic_sm_partition);
  STREAM_OUTPUT(colocate_config.skip_malloc);
  STREAM_OUTPUT(colocate_config.skip_loading);
  std::cerr << std::string(header.size(), '=') << std::endl;
}

void Shutdown(int sig) {
  LOG(INFO) <<"signal " <<  strsignal(sig) << "(" << sig << ")" << " received, shutting down...";
  colserve::Config::running = false;
  colserve::InferModelStore::Shutdown();
  colserve::TrainLauncher::Shutdown();
  colserve::Profiler::Shutdown();
  COL_NVML_CALL(nvmlShutdown());
  std::terminate();
}

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  colserve::Config::binary_directory = 
      std::filesystem::path(argv[0]).parent_path().parent_path();
  
  init_cli_options();
  CLI11_PARSE(app, argc, argv);
  init_config();

  std::thread shutdown_trigger([](){
    std::this_thread::sleep_for(std::chrono::minutes(max_live_minute));
    LOG(INFO) << "max live minute reached, shutting down...";
    Shutdown(SIGINT);
  });

  CHECK_EQ(cuInit(0), CUDA_SUCCESS);
  COL_NVML_CALL(nvmlInit());
  colserve::sta::DeviceManager::Init();
  auto free_list_policy = colserve::sta::getFreeListPolicy(
      colserve::Config::mempool_freelist_policy);
  if (colserve::Config::use_shared_tensor) {
    for (int device_id = 0; 
         device_id < colserve::sta::DeviceManager::GetNumVisibleGpu(); 
         device_id++) {
      colserve::sta::CUDAMemPool::Init(device_id,
        static_cast<size_t>(colserve::Config::cuda_memory_pool_gb * 1_GB),
        true, false, free_list_policy);
      colserve::sta::CUDAMemPool::Get(device_id)->RegisterOOMHandler([]() {
        LOG(INFO) << "train predict memory " 
                  <<  colserve::TrainAdjuster::PredictTrainMemUsageMB(0, true) 
                  << "."; 
        }, colserve::sta::MemType::kInfer);
    }
  }
  colserve::ctrl::Controller::Init();
  if (colserve::Config::dynamic_sm_partition) {
    for (int device_id = 0; 
         device_id < colserve::sta::DeviceManager::GetNumVisibleGpu(); 
         device_id++) {
      colserve::SMPartitioner::Init(device_id);
    }
  }
  colserve::ResourceManager::Init();
  colserve::Profiler::Init(colserve::Config::profile_log_path);
  colserve::TrainLauncher::Init("train");
  colserve::TrainAdjuster::Init();
  colserve::InferModelStore::Init("server/models");
  colserve::Profiler::Start();

  if (colserve::Config::memory_pressure_mb > 0) { 
    size_t nbytes = static_cast<size_t>(colserve::Config::memory_pressure_mb * 1_MB);
    COL_CUDA_CALL(cudaMalloc(&memory_pressure_ptr, nbytes));
  }

  std::string server_address("0.0.0.0:" + port);
  colserve::network::GRPCServer server;
  server.Start(server_address);

  std::signal(SIGINT, Shutdown);
  colserve::Config::system_initialized = true;

  server.Stop();
  LOG(INFO) << "server has shotdown";
}