#include <algorithm>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <regex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "colserve.grpc.pb.h"
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

  if (app.delay_before_infer > 0) {
    auto new_duration = app.duration + app.delay_before_infer;
    LOG(INFO) << "Override duration from " << app.duration << " to " << new_duration << ".";
    app.duration = new_duration;
  }


  std::string target = "localhost:" + app.port;
  colserve::workload::Workload workload(
      grpc::CreateChannel(target, grpc::InsecureChannelCredentials()),
      std::chrono::seconds(app.duration)
  );
  CHECK(workload.Hello());

  if (app.enable_infer && !app.infer_models.empty()) {
    for(auto &model : app.infer_models) {
      if (app.warmup > 0)
        workload.WarmupModel(model, app.warmup);
      workload.InferBusyLoop(model, app.concurrency, nullptr, app.delay_before_infer, app.show_result);
    }
    if (app.warmup > 0 && app.delay_after_warmup > 0) {
      std::this_thread::sleep_for(std::chrono::duration<double>(app.delay_after_warmup));
    }
  }
  
  if (app.enable_train) {
    if (app.train_models.count("resnet"))
      workload.TrainResnet(app.num_epoch, app.batch_size);
  }

  workload.Run();

  LOG(INFO) << "report result ...";
  if (app.log.empty()) {
    workload.Report(app.verbose);
  } else {
    std::fstream ofs{app.log, std::ios::out};
    CHECK(ofs.good());
    workload.Report(app.verbose, ofs);
    ofs.close();
  }
  return 0;
}