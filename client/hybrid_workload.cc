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
    app.add_option("--delay-before-infer", delay_before_infer, "delay before start infer.");
  }
};

struct InferModel {
  std::string model_name;
  // std::string type;
  std::string id;

  operator std::string() const {
    if (id.empty())
      return model_name;
    else
      return model_name + "-"+ id;
  }
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
  if (app.enable_infer) {
    trace_cfg = LoadTraceCFG(app.infer_trace);
    min_duration = trace_cfg.start_points.back().first + app.delay_before_infer  +3;
  }
  
  if (app.duration < min_duration) {
    LOG(INFO) << "Override duration from " << app.duration << " to " << min_duration << ".";
    app.duration = min_duration;
  }

  std::string target = "localhost:" + app.port;
  colserve::workload::Workload workload(
      grpc::CreateChannel(target, grpc::InsecureChannelCredentials()),
      std::chrono::seconds(app.duration)
  );
  CHECK(workload.Hello());


  if (app.enable_infer) {
    auto groups = GroupByModel(trace_cfg);
    for(auto &&[model_id, start_points] : groups) {
      auto &model = trace_cfg.models[model_id];
      workload.Infer(model.model_name, app.concurrency, start_points, app.delay_before_infer, app.show_result);
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