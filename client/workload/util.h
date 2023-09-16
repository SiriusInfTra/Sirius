#ifndef COLSYS_WORKLOAD_UTIL_H_
#define COLSYS_WORKLOAD_UTIL_H_

#include <iostream>
#include <fstream>
#include <CLI/CLI.hpp>
#include <glog/logging.h>

namespace colserve {
namespace workload {

std::string ReadInput(const std::string &data_path);

class AppBase {
 public:
  AppBase(const std::string &name = "ColServe Workload");

  CLI::App app;
  std::string port;
  bool enable_train{true}, enable_infer{true};
  std::set<std::string> infer_models, train_models;
  int duration{10}, concurrency{10};
  int num_epoch{1}, batch_size{1};
  
  std::string log;
  int verbose{0};
  uint32_t show_result{0};
};

}
}


#endif