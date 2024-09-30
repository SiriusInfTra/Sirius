#include <algorithm>
#include <fstream>
#include <future>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <grpcpp/grpcpp.h>
#include <grpcpp/channel.h>
#include <grpcpp/support/status.h>

#include "workload/util.h"
#include "workload/workload.h"
// #include "workload/workload.h"

using namespace colserve::workload;

struct App : public colserve::workload::AppBase {
  App(): colserve::workload::AppBase() {
    app.add_option("--infer-trace", infer_trace, "models of infer workload");
  }

  std::string infer_trace;
};

struct TraceCFG {
  std::unordered_map<std::string, InferModel> models;
  std::vector<std::pair<double, std::string>> start_points;
};


TraceCFG LoadTraceCFG(const std::string &trace_path) {
  std::ifstream handle;
  handle.open(trace_path);
  CHECK(handle.is_open());
  std::string line;
  std::getline(handle, line);
  CHECK_EQ(line, "# model_id,model_name");
  TraceCFG trace_cfg;
  do {
    std::getline(handle, line);
    std::istringstream iss{line};
    std::string model_id, model_name;
    std::getline(iss, model_id, ',');
    std::getline(iss, model_name, ',');
    trace_cfg.models[model_id] = InferModel{model_name, model_id};
  } while(line[0] != '#');
  CHECK_EQ(line, "# start_point,model_id");
  while(std::getline(handle, line)) {
    std::istringstream iss{line};
    std::string model_id, start_point;
    std::getline(iss, start_point, ',');
    std::getline(iss, model_id, ',');
    trace_cfg.start_points.emplace_back(std::stod(start_point), model_id);
    CHECK(trace_cfg.models.find(model_id) != trace_cfg.models.cend());
  }
  return trace_cfg;
  
}

std::vector<std::pair<std::string, std::vector<double>>> GroupByModel(const TraceCFG &trace_cfg) {
  std::unordered_map<std::string, std::vector<double>> groups;
  for(auto && [start_point, model_id] : trace_cfg.start_points) {
    groups[model_id].push_back(start_point);
  }
  std::vector<std::pair<std::string, std::vector<double>>> ret;
  std::transform(groups.cbegin(), groups.cend(), std::back_inserter(ret), [](auto &&pair){
    return std::move(pair);
  });
  return ret;
}


int main(int argc, char** argv) {
  App app;
  CLI11_PARSE(app.app, argc, argv);

  if (app.seed == static_cast<uint64_t>(-1)) {
    std::random_device rd;
    app.seed = rd();
  }
  LOG(INFO) << "Workload random seed " << app.seed;

  TraceCFG trace_cfg;
  double min_duration = -std::numeric_limits<double>::infinity();
  if (app.enable_infer && !app.infer_trace.empty()) {
    trace_cfg = LoadTraceCFG(app.infer_trace);
    min_duration = trace_cfg.start_points.back().first 
                   + app.wait_train_setup_sec 
                   + app.wait_stable_before_start_profiling_sec
                   + 3;
  }
  
  if (app.duration < min_duration) {
    LOG(INFO) << "Override duration from " << app.duration << " to " << min_duration << ".";
    app.duration = min_duration;
  }

  std::string colsys_target = app.colsys_ip + ":" + app.colsys_port;
  std::string triton_target = app.triton_ip + ":" + app.triton_port;

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
      app.infer_timeline
    );
  }
  // colserve::workload
  CHECK(train_workload == nullptr || train_workload->Hello());
  CHECK(infer_workload == nullptr || train_workload == infer_workload || infer_workload->Hello());

  if (app.enable_infer && !app.infer_trace.empty()) {
    auto groups = GroupByModel(trace_cfg);
    if (app.warmup > 0) {
      std::vector<std::future<void>> warm_up_futures;
      warm_up_futures.reserve(groups.size());
      for (auto &[model_id, _] : groups) {
        auto &model = trace_cfg.models[model_id];
        warm_up_futures.emplace_back(std::async(std::launch::async, 
            [&infer_workload, &model, &app](){
              infer_workload->WarmupModel(model.model_name, app.warmup);
            }
        ));
      }
      for (auto &f : warm_up_futures) {
        f.wait();
      }
      if (app.wait_warmup_done_sec > 0) {
        std::this_thread::sleep_for(std::chrono::duration<double>(app.wait_warmup_done_sec));
        infer_workload->WarmupDone();
      }
    }

    for(auto &&[model_id, start_points] : groups) {
      auto &model = trace_cfg.models[model_id];
      infer_workload->InferTrace(model.model_name, app.concurrency, 
                          start_points, app.wait_train_setup_sec,
                          app.warmup, app.show_result);
    }
  }
  
  if (app.enable_train) {
    // if (app.train_models.count("resnet152"))
    //   workload.TrainResnet(app.num_epoch, app.batch_size);
    CHECK(app.train_models.size() <= 1);
    if (!app.train_models.empty()) {
      auto model_name = *app.train_models.begin();
      train_workload->Train(model_name, app.num_epoch, app.batch_size);
    }
    train_workload->Run();
  }


  // TODO: merge output
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