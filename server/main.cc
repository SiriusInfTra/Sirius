#include <server/logging_as_glog.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/logging.h>

#include <grpcpp/grpcpp.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/health_check_service_interface.h>
#include <glog/logging.h>
#include <CLI/CLI.hpp>

#include <server/grpc/grpc_server.h>
#include <server/model_store/infer_model_store.h>
#include <server/llm/llm.h>
#include <server/train_launcher.h>
#include <server/train_adjuster.h>
#include <server/resource_manager.h>
#include <server/control/controller.h>
#include <server/profiler.h>
#include <server/config.h>

#include <common/cuda_allocator.h>
#include <common/sm_partition.h>
#include <common/device_manager.h>
#include <common/util.h>

#include <boost/range/irange.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/numeric.hpp>
#include <iostream>
#include <filesystem>
#include <csignal>

#define STREAM_OUTPUT(field) std::cerr << "cfg::" #field "=" << cfg::field << std::endl

#define READ_ENV_BOOL_CONFIG(env, field) do { \
    auto env_val = getenv(env); \
    if (env_val != nullptr) { \
      cfg::field = std::string(env_val) == "1"; \
    } \
  } while (false);

CLI::App app{"ColServe"};
std::string mode = "normal";
int max_live_minute = 20;

void* memory_pressure_ptr = nullptr;

void init_cli_options() {
  app.add_option("-m,--mode", mode,
      str(boost::format("server mode, see detail in server/config.h, default is %s") % mode))
      ->check(CLI::IsMember({"normal", 
                             "task-switch-l1", 
                             "task-switch-l2", 
                             "task-switch-l3",
                             "colocate-l1",
                             "colocate-l2"}));
  app.add_option("--mps", colserve::Config::check_mps, 
      str(boost::format("check mps, default is %d") 
          % colserve::Config::check_mps));
  app.add_option("--use-sta", 
      colserve::Config::use_shared_tensor, 
      str(boost::format("use shared tensor allocator, default is %d") 
          % colserve::Config::use_shared_tensor));
  app.add_option("--no-infer", 
      colserve::Config::no_infer, 
      "no infer, default is 0");
  app.add_option("--use-sta-infer", 
      colserve::Config::use_shared_tensor_infer, 
      str(boost::format("use shared tensor allocator in infer, default is %d") 
          % colserve::Config::use_shared_tensor_infer));
  app.add_option("--use-sta-train", 
      colserve::Config::use_shared_tensor_train,
      str(boost::format("use shared tensor allocator in train, default is %d") 
          % colserve::Config::use_shared_tensor_train));
  app.add_option("--cuda-memory-pool-gb", 
      colserve::Config::cuda_memory_pool_gb,
      str(boost::format("cuda memory pool size in GB, default is %.0f") 
          % colserve::Config::cuda_memory_pool_gb));
  app.add_option("--memory-pool-policy", 
      colserve::Config::mempool_freelist_policy, 
      str(boost::format("cuda memory pool freelist policy, default is %s") 
          % colserve::Config::mempool_freelist_policy))
        ->check(CLI::IsMember({"first-fit", "next-fit", "best-fit"}));
  app.add_option("-p,--port", colserve::Config::port,
      str(boost::format("gRPC server port, default is %s") % colserve::Config::port));
  app.add_option("--max-live-minute", max_live_minute,
      "max server live minute, default is " 
      + std::to_string(max_live_minute) + " minutes");
  app.add_option("--infer-model-config", 
      colserve::Config::infer_model_config_path, 
      str(boost::format("infer model config path, default is %s") 
          % colserve::Config::infer_model_config_path));
  app.add_option("--profile-log", 
      colserve::Config::profile_log_path, 
      str(boost::format("profile log path, default is %s") 
          % colserve::Config::profile_log_path));
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
      str(boost::format("capture train log, default is %d") 
          % colserve::Config::capture_train_log));
  app.add_flag("--infer-blob-alloc", 
      colserve::Config::infer_raw_blob_alloc, 
      str(boost::format("infer raw blob alloc, default is %d") 
          % colserve::Config::infer_raw_blob_alloc));
  app.add_option("--train-mps-thread-percent", 
      colserve::Config::train_mps_thread_percent, 
      str(boost::format("train mps thread percent, default is %d") 
          % colserve::Config::train_mps_thread_percent));
  app.add_flag("--colocate-skip-malloc", 
      colserve::Config::colocate_config.skip_malloc, 
      str(boost::format("colocate skip malloc, default is %d") 
          % colserve::Config::colocate_config.skip_malloc));
  app.add_flag("--colocate-skip-loading", 
      colserve::Config::colocate_config.skip_loading, 
      str(boost::format("colocate skip loading, default is %d") 
          % colserve::Config::colocate_config.skip_loading));
  app.add_option("--use-xsched", colserve::Config::use_xsched, 
      "use xsched, default is false");
  app.add_option("--dynamic-sm-partition", 
      colserve::Config::dynamic_sm_partition, 
      str(boost::format("dynamic sm partition, default is %d") 
          % colserve::Config::dynamic_sm_partition));
  app.add_option("--train-profile", colserve::Config::train_profile, 
      str(boost::format("train timeline path, default is %s") 
          % colserve::Config::train_profile));
  app.add_option("--max-warm-cache-nbytes", 
      colserve::Config::max_warm_cache_nbytes, 
      str(boost::format("max warm cache nbytes, default is %s") 
          % colserve::sta::PrintByte(colserve::Config::max_warm_cache_nbytes, true)));
  app.add_option("--cold-cache-min-capacity-nbytes", 
      colserve::Config::cold_cache_min_capacity_nbytes, 
      str(boost::format("min cold cache capacity nbytes, default is %s") 
          % colserve::sta::PrintByte(colserve::Config::cold_cache_min_capacity_nbytes, true)));
  app.add_option("--cold-cache-max-capacity-nbytes", 
      colserve::Config::cold_cache_max_capacity_nbytes, 
      str(boost::format("max cold cache capacity nbytes, default is %s") 
          % colserve::sta::PrintByte(colserve::Config::cold_cache_max_capacity_nbytes, true)));
  app.add_option("--cold-cache-ratio", 
      colserve::Config::cold_cache_ratio, 
      str(boost::format("cold cache ratio, default is %.1f(%.0f%%)") 
          % colserve::Config::cold_cache_ratio 
          % (colserve::Config::cold_cache_ratio * 100)));
  app.add_option("--memory-pressure-mb", 
      colserve::Config::memory_pressure_mb,
      str(boost::format("memory pressure in MB, default is %.0f") 
          % colserve::Config::memory_pressure_mb));
  app.add_option("--ondemand-adjust", colserve::Config::ondemand_adjust,
      str(boost::format("ondemand adjust batch size, default is %d") 
          % colserve::Config::ondemand_adjust));
  app.add_option("--pipeline-load", colserve::Config::pipeline_load,
      str(boost::format("pipeline load, default is %d") 
          % colserve::Config::pipeline_load));
  app.add_option("--train-memory-over-predict-mb", 
      colserve::Config::train_memory_over_predict_mb,
      str(boost::format("train memory over predict in MB, default is %.0f") 
          % colserve::Config::train_memory_over_predict_mb));
  app.add_option("--infer-model-max-idle-ms", 
      colserve::Config::infer_model_max_idle_ms,
      str(boost::format("infer model max idle in ms, default is %.0f") 
          % colserve::Config::infer_model_max_idle_ms));
  app.add_option("--has-warmup", colserve::Config::has_warmup,
      str(boost::format("has warmup, default is %d") 
      % colserve::Config::has_warmup));
  app.add_option("--train-adjust-balance", colserve::Config::enable_train_adjust_balance,
      str(boost::format("train adjust balance, default is %d") 
          % colserve::Config::enable_train_adjust_balance));
  app.add_option("--train-adjust-batch-size-limit", 
      colserve::Config::train_adjust_batch_size_limit,
      str(boost::format("train adjust batch size limit, default is %d") 
          % colserve::Config::train_adjust_batch_size_limit));
  app.add_flag("--serving-llm", colserve::Config::serving_llm,
      str(boost::format("serving llm, default is %d") 
          % colserve::Config::serving_llm));
  app.add_option("--llm-model-name", colserve::Config::llm_model_name,
      str(boost::format("llm model name, default is %s") 
          % colserve::Config::llm_model_name));
  app.add_option("--llm-max-seq-len", colserve::Config::llm_max_seq_len,
      str(boost::format("llm max seq len, default is %d") 
          % colserve::Config::llm_max_seq_len));
  app.add_option("--llm-max-model-len", colserve::Config::llm_max_model_len,
      str(boost::format("llm max model len, default is %d") 
          % colserve::Config::llm_max_model_len));
  app.add_flag("--llm-show-gen-result", colserve::Config::llm_show_gen_result,
      str(boost::format("llm show gen result, default is %d") 
          % colserve::Config::llm_show_gen_result));
  app.add_option("--llm-show-gen-result-period", colserve::Config::llm_show_gen_result_period,
      str(boost::format("llm show gen result period, default is %d") 
          % colserve::Config::llm_show_gen_result_period));
  app.add_flag("--dump-adjust-info", 
      colserve::Config::dump_adjust_info,
      str(boost::format("dump adjust info, default is %d") 
          % colserve::Config::dump_adjust_info));
  app.add_flag("--profiler-acquire-resource-lock", 
      colserve::Config::profiler_acquire_resource_lock,
      str(boost::format("profiler acquire resource lock during profiling, "
                        "not use for performance eval, default is %d") 
          % colserve::Config::profiler_acquire_resource_lock));
  app.add_flag("--dummy-adjust", colserve::Config::dummy_adjust,
      str(boost::format("dummy adjust for eval, default is %d") 
          % colserve::Config::dummy_adjust));
  app.add_flag("--skip-set-mps-thread-percent", 
      colserve::Config::skip_set_mps_thread_percent,
      str(boost::format("skip set mps thread percent, default is %d") 
          % colserve::Config::skip_set_mps_thread_percent));

  app.add_flag("--enable-warm-cache-fallback", 
      colserve::Config::enable_warm_cache_fallback,
      str(boost::format("enable warm cache fallback, default is %d") 
          % colserve::Config::enable_warm_cache_fallback));

  app.footer(
      "The following environment variables can be used to configure the server logging:\n"
      "  COLSERVE_LOG_ALL,                 COLSERVE_LOG_MEMORY_ADJUST,       COLSERVE_LOG_GRPC\n"
      "  COLSERVE_LOG_TRAIN_INIT,          COLSERVE_LOG_WARM_CACHE,          COLSERVE_LOG_COLD_CACHE\n"
      "  COLSERVE_LOG_INFER_MODEL_INIT,    COLSERVE_LOG_INFER_MODEL_RECLAIM, COLSERVE_LOG_INFER_TIME\n"
      "  COLSERVE_LOG_INFER_PIPELINE_EXEC, COLSERVE_LOG_INFER_LOAD_PARAM,    COLSERVE_LOG_CONTROLLER\n"
      "  COLSERVE_LOG_TASK_SWITCH\n\n"
      "Environment variables that used to configure the training logging:\n"
      "  COLTRAIN_LOG_ALL                  COLTRAIN_LOG_DYNAMIC_BATCH\n"
      "  COLTRAIN_LOG_CONTROL_STUB         COLTRAIN_LOG_NCCL_PROCESS_GROUP\n"
  );
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
                           - cfg::cold_cache_max_capacity_nbytes); // for warmup
    LOG(INFO) << "enable enable_warm_cache_fallback, "
              << "cache nbytes (used in warmup, conservative estimated) " 
              << colserve::sta::PrintByte(cfg::max_warm_cache_nbytes, true);
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
  auto col_log_adjust_env = getenv("COLSERVE_LOG_MEMORY_ADJUST");
  if (col_log_adjust_env != nullptr) {
    cfg::log_memory_adjust = std::string(col_log_adjust_env) == "1";
  }

  // disable dcgm in docker
  if (std::filesystem::exists("/.dockerenv")) {
    cfg::profile_gpu_smact = false;
    cfg::profile_gpu_util = false;
  }

  READ_ENV_BOOL_CONFIG("COLSERVE_LOG_GRPC", log_grpc);
  READ_ENV_BOOL_CONFIG("COLSERVE_LOG_TRAIN_INIT", log_train_init);
  READ_ENV_BOOL_CONFIG("COLSERVE_LOG_WARM_CACHE", log_warm_cache);
  READ_ENV_BOOL_CONFIG("COLSERVE_LOG_COLD_CACHE", log_cold_cache);
  READ_ENV_BOOL_CONFIG("COLSERVE_LOG_INFER_MODEL_INIT", log_infer_model_init);
  READ_ENV_BOOL_CONFIG("COLSERVE_LOG_INFER_MODEL_RECLAIM", log_infer_model_reclaim);
  READ_ENV_BOOL_CONFIG("COLSERVE_LOG_INFER_TIME", log_infer_time);
  READ_ENV_BOOL_CONFIG("COLSERVE_LOG_INFER_PIPELINE_EXEC", log_infer_pipeline_exec);
  READ_ENV_BOOL_CONFIG("COLSERVE_LOG_INFER_LOAD_PARAM", log_infer_load_param);
  READ_ENV_BOOL_CONFIG("COLSERVE_LOG_CONTROLLER", log_controller);
  READ_ENV_BOOL_CONFIG("COLSERVE_LOG_TASK_SWITCH", log_task_switch);

  CHECK(setenv("SIRIUS_PORT", cfg::port.c_str(), 1) == 0);

  if (cfg::serving_llm) {
    if (cfg::enable_train_adjust_balance) {
      LOG(WARNING) << "Serving LLM, currently disable train adjust balance";
      cfg::enable_train_adjust_balance = false;
    }
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
    cfg::log_task_switch = true;
  }

std::string header = str(boost::format("%s COLSERVE CONFIG [PID %d | PORT %d] %s") 
                          % std::string(16, '=')
                          % getpid()
                          % std::stoi(cfg::port)
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
  STREAM_OUTPUT(train_memory_over_predict_mb);
  STREAM_OUTPUT(cold_cache_max_capacity_nbytes);
  STREAM_OUTPUT(cold_cache_min_capacity_nbytes);
  STREAM_OUTPUT(train_over_adjust_nbytes);
  STREAM_OUTPUT(cold_cache_ratio);
  STREAM_OUTPUT(infer_model_max_idle_ms);
  STREAM_OUTPUT(dynamic_sm_partition);
  STREAM_OUTPUT(enable_train_adjust_balance);
  STREAM_OUTPUT(train_adjust_balance_threshold);
  STREAM_OUTPUT(serving_llm);
  STREAM_OUTPUT(llm_model_name);
  STREAM_OUTPUT(llm_max_model_len);
  STREAM_OUTPUT(llm_max_seq_len);
  STREAM_OUTPUT(colocate_config.skip_malloc);
  STREAM_OUTPUT(colocate_config.skip_loading);
  std::cerr << std::string(header.size(), '=') << std::endl;
}

void Shutdown(int sig) {
  // FIXME: flush pytorc output
  LOG(INFO) <<"signal " <<  strsignal(sig) << "(" << sig << ")" 
            << " received, shutting down...";
  colserve::Config::running = false;
  if (!colserve::Config::serving_llm) {
    colserve::InferModelStore::Shutdown();
  } else {
    colserve::LLMServer::Shutdown();
  }
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
    LOG(INFO) << "max live minute (" << max_live_minute
              << ") reached, shutting down...";
    Shutdown(SIGINT);
  });
  if (!colserve::Config::no_infer) {
    COL_CU_CALL(cuInit(0));
  }
  COL_NVML_CALL(nvmlInit());
  colserve::sta::DeviceManager::Init();
  auto free_list_policy = colserve::sta::getFreeListPolicy(
      colserve::Config::mempool_freelist_policy);
  if (colserve::Config::use_shared_tensor && !colserve::Config::no_infer) {
    for (int device_id = 0; 
         device_id < colserve::sta::DeviceManager::GetNumVisibleGpu(); 
         device_id++) {
      colserve::sta::CUDAMemPool::Init(device_id,
        static_cast<size_t>(colserve::Config::cuda_memory_pool_gb * 1_GB),
        true, false, free_list_policy, true);
      colserve::sta::CUDAMemPool::Get(device_id)->RegisterOOMHandler([]() {
        LOG(INFO) << "[CUDAMemPool INFER OOM] train predict memory" 
                  << boost::accumulate(
                      boost::irange(colserve::sta::DeviceManager::GetNumVisibleGpu()), 
                      std::string{""}, [](std::string acc, int device_id) {
                        return acc + " " + std::to_string(
                            colserve::TrainAdjuster::PredictTrainMemUsageMB(device_id, true));
                      }) 
                  << ".";
        }, colserve::sta::MemType::kInfer);
        colserve::sta::CUDAMemPool::Get(device_id)->RegisterOOMHandler([]() {
        LOG(INFO) << "[CUDAMemPool TRAIN OOM] train predict memory" 
                  << boost::accumulate(
                      boost::irange(colserve::sta::DeviceManager::GetNumVisibleGpu()), 
                      std::string{""}, [](std::string acc, int device_id) {
                        return acc + " " + std::to_string(
                            colserve::TrainAdjuster::PredictTrainMemUsageMB(device_id, true));
                      }) 
                  << ".";
        }, colserve::sta::MemType::kTrain);
    }
  } else {
    for (int device_id = 0; 
         device_id < colserve::sta::DeviceManager::GetNumVisibleGpu(); 
         device_id++) {
      colserve::sta::CUDAMemPool::Init(device_id,
        static_cast<size_t>(colserve::Config::cuda_memory_pool_gb * 1_GB),
        false, false, free_list_policy, false);
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
  if (!colserve::Config::serving_llm) {
    colserve::InferModelStore::Init("server/models");
  } else {
    colserve::LLMServer::Init();
  }
  colserve::Profiler::Start();

  if (colserve::Config::memory_pressure_mb > 0) { 
    size_t nbytes = static_cast<size_t>(colserve::Config::memory_pressure_mb * 1_MB);
    COL_CUDA_CALL(cudaMalloc(&memory_pressure_ptr, nbytes));
  }

  std::string server_address("0.0.0.0:" + colserve::Config::port);
  colserve::network::GRPCServer server;
  server.Start(server_address);

  std::signal(SIGINT, Shutdown);
  colserve::Config::system_initialized = true;

  server.Stop();
  LOG(INFO) << "server has shutdown";
}