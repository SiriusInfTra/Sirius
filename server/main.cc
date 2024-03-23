#include "logging_as_glog.h"
#include "common/cuda_allocator.h"
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

#include <common/init.h>
#include <common/mempool.h>
#include <common/util.h>

#include "grpc/grpc_server.h"
#include "colserve.grpc.pb.h"
#include "infer_model_store.h"
#include "train_launcher.h"
#include "cache.h"
#include "resource_manager.h"
#include "controller.h"
#include "profiler.h"
#include "config.h"

CLI::App app{"ColServe"};
std::string mode = "normal";
std::string port = "8080";
int max_live_minute = 15;

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
  app.add_option("--use-sta", colserve::Config::use_shared_tensor, 
      "use shared tensor allocator, default is 1");
  app.add_option("--use-sta-infer", colserve::Config::use_shared_tensor_infer, 
      "use shared tensor allocator in infer, default is 1");
  app.add_option("--use-sta-train", colserve::Config::use_shared_tensor_train,
      "use shared tensor allocator in train, default is 1");
  app.add_option("--cuda-memory-pool-gb", colserve::Config::cuda_memory_pool_gb,
      "cuda memory pool size in GB, default is 12");
  app.add_option("--memory-pool-policy", colserve::Config::mempool_freelist_policy, 
        "cuda memory pool freelist policy, default is best-fit.")
        ->check(CLI::IsMember({"first-fit", "next-fit", "best-fit"}));
  app.add_option("-p,--port", port,
      "gRPC server port, default is 8080");
  app.add_option("--max-live-minute", max_live_minute,
      "max server live minute, default is " + std::to_string(max_live_minute) + " minutes");
  app.add_option("--infer-model-config", colserve::Config::infer_model_config_path, 
      "infer model config path, default is config in models store");
  app.add_option("--profile-log", colserve::Config::profile_log_path, 
      "profile log path, default is server-profile");
  app.add_option("--capture-train-log", colserve::Config::capture_train_log, 
      "capture train log, default is 1");
  app.add_flag("--infer-blob-alloc", colserve::Config::infer_raw_blob_alloc, 
      "infer raw blob alloc, default is false");
  app.add_option("--train-mps-thread-percent", colserve::Config::train_mps_thread_percent, 
      "train mps thread percent, default is None");
  app.add_flag("--colocate-skip-malloc", colserve::Config::colocate_config.skip_malloc, 
      "colocate skip malloc, default is false");
  app.add_flag("--colocate-skip-loading", colserve::Config::colocate_config.skip_loading, 
      "colocate skip loading, default is false");
  app.add_option("--use-xsched", colserve::Config::use_xsched, 
      "use xsched, default is false");
  app.add_option("--train-profile", colserve::Config::train_profile, 
    "train timeline path, default is train-timeline");
  app.add_option("--max-cache-nbytes", colserve::Config::max_cache_nbytes, 
    "max cache nbytes, default is 1*1024*1024*1024(1G).");
  app.add_option("--memory-pressure-mb", colserve::Config::memory_pressure_mb,
      "memory pressure in MB, default is 0");
  app.add_option("--ondemand-adjust", colserve::Config::ondemand_adjust,
      "ondemand adjust batch size, default is 1");
  app.add_option("--pipeline-load", colserve::Config::pipeline_load,
      "pipeline load, default is 1");
  app.add_option("--train-memory-over-predict-mb", colserve::Config::train_memory_over_predict_mb,
      "train memory over predict in MB, default is 2560");
  app.add_option("--infer-model-max-idle-ms", colserve::Config::infer_model_max_idle_ms,
      "infer model max idle in ms, default is 3000");
  app.add_option("--has-warmup", colserve::Config::has_warmup,
      "has warmup, default is 0");
  app.add_flag("--dummy-adjust", colserve::Config::dummy_adjust,
      "dummy adjust for eval, default is 0");
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
      colserve::Config::use_shared_tensor_infer && colserve::Config::use_shared_tensor;
  colserve::Config::use_shared_tensor_train =
      colserve::Config::use_shared_tensor_train && colserve::Config::use_shared_tensor;

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

  std::cerr << "cfg::serve_mode=" << static_cast<int>(cfg::serve_mode) << std::endl
            << "cfg::use_shared_tensor=" << cfg::use_shared_tensor << std::endl
            << "cfg::use_shared_tensor_infer=" << cfg::use_shared_tensor_infer << std::endl
            << "cfg::use_shared_tensor_train=" << cfg::use_shared_tensor_train << std::endl
            << "cfg::ondemand_adjust=" << cfg::ondemand_adjust << std::endl
            << "cfg::better_alloc=" << cfg::better_alloc << std::endl
            << "cfg::group_param_load=" << cfg::group_param_load << std::endl
            << "cfg::pipeline_load=" << cfg::pipeline_load << std::endl
            << "cfg::has_warmup=" << cfg::has_warmup << std::endl
            << "cfg::colocate_config.skip_malloc=" << cfg::colocate_config.skip_malloc << std::endl
            << "cfg::colocate_config.skip_loading=" << cfg::colocate_config.skip_loading << std::endl;

}

void Shutdown(int sig) {
  LOG(INFO) <<"signal " <<  strsignal(sig) << " received, shutting down...";
  colserve::Config::running = false;
  colserve::InferModelStore::Shutdown();
  colserve::TrainLauncher::Shutdown();
  colserve::Profiler::Shutdown();
  std::terminate();
}

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  colserve::Config::binary_directory = std::filesystem::path(argv[0]).parent_path().parent_path();
  
  init_cli_options();
  CLI11_PARSE(app, argc, argv);
  init_config();

  std::thread shutdown_trigger([](){
    std::this_thread::sleep_for(std::chrono::minutes(max_live_minute));
    LOG(INFO) << "max live minute reached, shutting down...";
    Shutdown(SIGINT);
  });

  CHECK_EQ(cuInit(0), CUDA_SUCCESS);
  auto free_list_policy = colserve::sta::getFreeListPolicy(
      colserve::Config::mempool_freelist_policy);
  if (colserve::Config::use_shared_tensor) {
    colserve::sta::InitMemoryPool(
      static_cast<size_t>(colserve::Config::cuda_memory_pool_gb * 1_GB),
      true, false, free_list_policy);
    colserve::sta::CUDAMemPool::Get()->RegisterOOMHandler([]() {
      LOG(INFO) << "train predict memory " 
                <<  colserve::TrainLauncher::Get()->PredictMemUsageMB() << "."; 
      }, colserve::sta::MemType::kInfer);
  }
  colserve::ResourceManager::Init();
  colserve::Controller::Init();
  colserve::Profiler::Init(colserve::Config::profile_log_path);
  colserve::GraphCache::Init(colserve::Config::max_cache_nbytes);
  colserve::TrainLauncher::Init("train");
  colserve::InferModelStore::Init("server/models");
  colserve::Profiler::Start();

  if (colserve::Config::memory_pressure_mb > 0) { 
    size_t nbytes = static_cast<size_t>(colserve::Config::memory_pressure_mb * 1_MB);
    CUDA_CALL(cudaMalloc(&memory_pressure_ptr, nbytes));
  }

  std::string server_address("0.0.0.0:" + port);
  colserve::network::GRPCServer server;
  server.Start(server_address);

  std::signal(SIGINT, Shutdown);
  colserve::Config::system_initialized = true;

  server.Stop();
  LOG(INFO) << "server has shotdown";
}