#ifndef COLSERVE_PROFILER_H
#define COLSERVE_PROFILER_H

#include <iostream>
#include <chrono>
#include <vector>
#include <thread>
#include <mutex>
#include <fstream>


namespace colserve {

class Profiler {
 public:
  using time_point_t = std::chrono::time_point<std::chrono::steady_clock>;
  
  static void Init(const std::string &profile_log_path);
  static void Start();
  static void Shutdown();
  static Profiler* Get();
  
  struct ResourceInfo {
    size_t infer_mem;
    size_t train_mem;
    size_t gpu_used_mem;
  };
  struct InferInfo {
    size_t id;
    double recv_time;
    double finish_time;
  };
  enum class EventItem {
    // AddInferStart,
    TrainAdjustStart,
    TrainAdjustEnd,
    InferAllocStorageStart,
    InferAllocStorageEnd,
    InferLoadParamStart,
    InferLoadParamEnd,
    AddInfer,
    InferExit,
  };
  
  Profiler(const std::string &profile_log_path);
  ~Profiler();
  void RecordEvent(EventItem item);
  
  friend std::ostream& operator<<(std::ostream &os, EventItem item);

 private:
  double Passed();
  void WriteLog();
  static std::unique_ptr<Profiler> profiler_;
  time_point_t stp_;
  std::atomic<bool> start_profile_;
  std::string profile_log_path_;

  std::vector<InferInfo> infer_info_;
  std::vector<std::tuple<double, ResourceInfo>> resource_info_;
  std::vector<std::tuple<double, EventItem>> event_info_;
  std::mutex infer_info_mut_, event_info_mut_;

  std::unique_ptr<std::thread> thread_;
};

}

#endif