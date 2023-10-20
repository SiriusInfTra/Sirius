#include <iostream>
#include <csignal>
#include "model_infer_store.h"
#include <grpcpp/grpcpp.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/health_check_service_interface.h>
#include <glog/logging.h>
#include <CLI/CLI.hpp>
#include <cuda.h>

#include <sta/init.h>

#include "grpc/grcp_server.h"
#include "colserve.grpc.pb.h"
#include "model_train_store.h"
#include "controller.h"
#include "profiler.h"
#include "config.h"

CLI::App app{"ColServe"};
std::string mode = "normal";
std::string port = "8080";
int max_live_minute = 15;

void init_cli_options() {
  app.add_option("-m,--mode", mode,
      "server mode, see detail in server/config.h, default is normal")
      ->check(CLI::IsMember({"normal", 
                             "task-switch-l1", 
                             "task-switch-l2", 
                             "task-switch-l3",
                             "colocate-l1",
                             "colocate-l2"}));
  app.add_option("--use-sta", colserve::Config::use_shared_tensor, 
      "use shared tensor allocator, default is 1");
  app.add_option("-p,--port", port,
      "gRPC server port, default is 8080");
  app.add_option("--max-live-minute", max_live_minute,
      "max server live minute, default is " + std::to_string(max_live_minute) + " minutes");
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
}

void Shutdown(int sig) {
  LOG(INFO) << "SIGINT received, shutting down...";
  colserve::Config::running = false;
  colserve::ModelInferStore::Shutdown();
  colserve::ModelTrainStore::Shutdown();
  colserve::Profiler::Shutdown();
  std::terminate();
}

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);

  init_cli_options();
  CLI11_PARSE(app, argc, argv);
  init_config();

  std::thread shutdown_trigger([](){
    std::this_thread::sleep_for(std::chrono::minutes(max_live_minute));
    LOG(INFO) << "max live minute reached, shutting down...";
    Shutdown(SIGINT);
  });

  CHECK_EQ(cuInit(0), CUDA_SUCCESS);
  if (colserve::Config::use_shared_tensor) {
    colserve::sta::Init(true);
  }
  colserve::Controller::Init();
  colserve::Profiler::Init("server-profile");
  colserve::ModelInferStore::Init("models");
  colserve::ModelTrainStore::Init("train");
  colserve::Profiler::Start();

  std::string server_address("0.0.0.0:" + port);
  colserve::network::GRPCServer server;
  server.Start(server_address);

  std::signal(SIGINT, Shutdown);

  server.Stop();
  LOG(INFO) << "server has shotdown";
}