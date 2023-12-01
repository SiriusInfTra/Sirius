#ifndef COLSYS_WORKLOAD_UTIL_H_
#define COLSYS_WORKLOAD_UTIL_H_

#include <condition_variable>
#include <iostream>
#include <fstream>
#include <CLI/CLI.hpp>
#include <limits>
#include <mutex>
#include <glog/logging.h>

namespace colserve {
namespace workload {

std::string ReadInput(const std::string &data_path);

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

class AppBase {
 public:
  AppBase(const std::string &name = "ColServe Workload");

  CLI::App app;
  std::string port;
  bool enable_train{true}, enable_infer{true};
  std::set<std::string> train_models;
  double delay_before_infer;
  int duration{10}, concurrency{10};
  int num_epoch{1}, batch_size{1};

  int warmup{10};
  double delay_after_warmup{0};

  std::string log;
  int verbose{0};
  int64_t show_result{0};

  static uint64_t seed;
};

}
}




#endif