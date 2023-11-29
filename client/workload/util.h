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

class AppBase {
 public:
  AppBase(const std::string &name = "ColServe Workload");

  CLI::App app;
  std::string port;
  bool enable_train{true}, enable_infer{true};
  std::string infer_trace;
  std::set<std::string> train_models;
  double delay_before_infer;
  int duration{10}, concurrency{10};
  int num_epoch{1}, batch_size{1};
  int warmup{10};

  std::string log;
  int verbose{0};
  int64_t show_result{0};

  static uint64_t seed;
};

std::vector<std::vector<unsigned>> load_azure(
    unsigned trace_id, unsigned period = std::numeric_limits<unsigned>::max());

class CountDownLatch {
public:
  explicit CountDownLatch(int count);
  void Reset(int count);
  void CountDownAndWait();
  void Wait();

 private:
  int count_;
  std::mutex mutex_;
  std::condition_variable condition_;
};
}
}




#endif