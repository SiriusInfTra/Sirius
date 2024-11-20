#include <cstddef>
#include <fstream>
#include <future>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <grpcpp/grpcpp.h>
#include <grpcpp/channel.h>
#include <grpcpp/support/status.h>
#include <colserve.pb.h>

#include "workload/util.h"
#include "workload/workload.h"

using namespace colserve::workload;

struct App : public colserve::workload::AppBase {
  App(): colserve::workload::AppBase() {
    app.add_option("--infer-model", infer_models, "models of infer workload");
  }

  std::set<std::string> infer_models;
};


int main(int argc, char** argv) {
  App app;
  CLI11_PARSE(app.app, argc, argv);

  if (app.seed == static_cast<uint64_t>(-1)) {
    std::random_device rd;
    app.seed = rd();
  }
  LOG(INFO) << "Workload random seed " << app.seed;

  if (app.wait_train_setup_sec > 0) {
    auto new_duration = app.duration
                        + app.wait_train_setup_sec
                        + app.wait_stable_before_start_profiling_sec;
    LOG(INFO) << "Override duration from " 
              << app.duration << " to " << new_duration << ".";
    app.duration = new_duration;
  }


  std::string colsys_target = app.colsys_ip + ":" + app.colsys_port;
  std::string triton_target = app.triton_ip + ":" + app.triton_port;
  LOG(INFO) << "Connect to " << colsys_target;
  std::shared_ptr<IWorkload> train_workload;
  if (!app.colsys_port.empty()) {
    LOG(INFO) << "Connect to " << colsys_target;
    train_workload = GetColsysWorkload(
      grpc::CreateChannel(colsys_target, grpc::InsecureChannelCredentials()),
      std::chrono::seconds(app.duration),
      app.wait_train_setup_sec + app.wait_stable_before_start_profiling_sec,
      app.infer_timeline
    );
  }
  std::shared_ptr<IWorkload> infer_workload;
  if (app.triton_port.empty()) {
    infer_workload = train_workload;
  } else {
    LOG(INFO) << "Connect to " << triton_target;
    infer_workload = GetTritonWorkload(
      grpc::CreateChannel(triton_target, grpc::InsecureChannelCredentials()),
      std::chrono::seconds(app.duration),
      app.wait_train_setup_sec + app.wait_stable_before_start_profiling_sec,
      app.infer_timeline, app.triton_max_memory, 
      app.triton_config, app.triton_device_map
    );
  }
  CHECK(train_workload == nullptr || train_workload->Hello());
  CHECK(infer_workload == nullptr 
        || train_workload == infer_workload 
        || infer_workload->Hello());

  if (app.enable_infer && !app.infer_models.empty()) {
    if (app.warmup > 0) {
      std::vector<std::future<void>> warm_up_futures;
      warm_up_futures.reserve(app.infer_models.size());
      for (auto &model : app.infer_models) {
        warm_up_futures.push_back(std::async(std::launch::async, 
            [&infer_workload, &model, &app](){
              infer_workload->WarmupModel(model, app.warmup);
            }
        ));
      }
      for (auto &f : warm_up_futures) {
        f.wait();
      }
      LOG(INFO) << "Warmup done.";
      if (app.wait_warmup_done_sec > 0) {
        std::this_thread::sleep_for(std::chrono::duration<double>(app.wait_warmup_done_sec));
        infer_workload-> WarmupDone();
      }
    }
    for(auto &model : app.infer_models) {
      infer_workload->InferBusyLoop(
          model, app.concurrency, nullptr, app.wait_train_setup_sec, 
          app.warmup, app.show_result);
    }
  }
  
  if (app.enable_train) {
    if (app.train_models.count("resnet"))
      train_workload->Train("resnet", app.num_epoch, app.batch_size);
  }

  if (infer_workload != nullptr) {
    infer_workload->PreRun();
  }
  if (train_workload != nullptr && infer_workload != train_workload) {
    train_workload->PreRun();
  }
  std::this_thread::sleep_for(std::chrono::seconds(app.duration));
  if (infer_workload != nullptr) {
    infer_workload->PostRun();
  }
  if (train_workload != nullptr && infer_workload != train_workload) {
    train_workload->PostRun();
  }

  LOG(INFO) << "report result ...";
  std::fstream fstream;
  auto &&ofs = app.log.empty() ? std::cout : (fstream = std::fstream{app.log, std::ios::out});
  CHECK(ofs.good());
  if (train_workload != nullptr) {
    train_workload->Report(app.verbose, ofs);
  }
  if (infer_workload != nullptr && train_workload != infer_workload) {
    infer_workload->Report(app.verbose, ofs);
  }
  if (fstream.is_open()) {
    fstream.close();
  }
  return 0;
}