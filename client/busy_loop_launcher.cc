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
    LOG(INFO) << "Override duration from " << app.duration << " to " << new_duration << ".";
    app.duration = new_duration;
  }


  std::string target = app.colsys_ip + ":" + app.colsys_port;
  auto workload = colserve::workload::GetColsysWorkload(
      grpc::CreateChannel(target, grpc::InsecureChannelCredentials()),
      std::chrono::seconds(app.duration),
      app.wait_train_setup_sec + app.wait_stable_before_start_profiling_sec,
      app.infer_timeline
  );
  CHECK(workload->Hello());

  if (app.enable_infer && !app.infer_models.empty()) {
    if (app.warmup > 0) {
      std::vector<std::future<void>> warm_up_futures;
      warm_up_futures.reserve(app.infer_models.size());
      for (auto &model : app.infer_models) {
        warm_up_futures.push_back(std::async(std::launch::async, 
            [&workload, &model, &app](){
              workload->WarmupModel(model, app.warmup);
            }
        ));
      }
      for (auto &f : warm_up_futures) {
        f.wait();
      }
      if (app.wait_warmup_done_sec > 0) {
        std::this_thread::sleep_for(std::chrono::duration<double>(app.wait_warmup_done_sec));
        workload-> WarmupDone();
      }
    }
    for(auto &model : app.infer_models) {
      workload-> InferBusyLoop(model, app.concurrency, nullptr, app.wait_train_setup_sec, 
                             app.warmup, app.show_result);
    }
  }
  
  if (app.enable_train) {
    if (app.train_models.count("resnet"))
      workload->Train("resnet", app.num_epoch, app.batch_size);
  }

  workload-> Run();

  LOG(INFO) << "report result ...";
  if (app.log.empty()) {
    workload->Report(app.verbose);
  } else {
    std::fstream ofs{app.log, std::ios::out};
    CHECK(ofs.good());
    workload->Report(app.verbose, ofs);
    ofs.close();
  }
  return 0;
}