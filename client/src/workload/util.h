#ifndef COLSYS_WORKLOAD_UTIL_H_
#define COLSYS_WORKLOAD_UTIL_H_

#include <CLI/CLI.hpp>
#include <glog/logging.h>

namespace colserve {
namespace workload {

std::string ReadInput(std::filesystem::path data_path);

int GetLLMMaxModelLen(const std::string &model_name);

inline long GetTimeStamp() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
}

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
  std::string colsys_ip = "localhost";
  std::string colsys_port;
  std::string triton_ip = "localhost";
  std::string triton_port;
  size_t triton_max_memory{0}; // in MB
  std::string triton_config;
  std::string triton_device_map;
  bool enable_train{true}, enable_infer{true};
  std::set<std::string> train_models;
  int duration{10}, concurrency{10};
  int num_epoch{1}, batch_size{1};

  int warmup{10};

  double wait_warmup_done_sec{0};
  double wait_train_setup_sec{0};
  double wait_stable_before_start_profiling_sec{0};

  /*
   * [Note: client timeline]
   *
   * warmup ---[wait warmup]--> workload start (record start timestamp) 
   *        ---[wait train]---> send infer req 
   *        ---[wait stable]--> start profiling ---> end
   * 
   * infer request delay after start = wait train setup
   * profiling delay after start = wait train setup + wait stable 
   */ 

  std::string log;
  int verbose{0};
  int64_t show_result{0};

  std::string infer_timeline;

  static uint64_t seed;
};

}
}




#endif