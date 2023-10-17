#ifndef COLSERVE_PROFILER_H
#define COLSERVE_PROFILER_H

#include <iostream>
#include <chrono>
#include <vector>
#include <thread>
#include <mutex>
#include <fstream>
#include <unordered_map>

namespace colserve {


class Profiler {
 public:
  using time_point_t = std::chrono::time_point<std::chrono::steady_clock>;
  
  static void Init(const std::string &profile_log_path);
  static void Start();
  static void Shutdown();
  inline static double Milli(time_point_t begin, time_point_t end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
  }
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
  enum class PerfItem {
    InferQueue,
    InferProcess,

    TrainAdjust,
    InferAllocStorage,
    InferLoadParam,
  };


  Profiler(const std::string &profile_log_path);
  ~Profiler();
  void RecordEvent(EventItem item);
  void RecordEvent(EventItem item, time_point_t tp);
  void RecordPerf(PerfItem item, double value);
  void RecordPerf(PerfItem item, time_point_t start, time_point_t end);
  
  friend std::ostream& operator<<(std::ostream &os, EventItem item);
  friend std::ostream& operator<<(std::ostream &os, PerfItem item);

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
  std::unordered_map<int, std::vector<double>> perf_info_;
  std::mutex infer_info_mut_, event_info_mut_, perf_info_mut_;

  std::unique_ptr<std::thread> thread_;
};

#define PROFILE_START(item, idx) \
    auto __t_ ## idx ## _ ## item ## _start_ = std::chrono::steady_clock::now(); 

#define PROFILE_END(item, idx) \
    auto __t_ ## idx ## _ ## item ## _end_ = std::chrono::steady_clock::now(); \
    Profiler::Get()->RecordEvent(Profiler::EventItem::item##Start, __t_ ## idx ## _ ## item ## _start_); \
    Profiler::Get()->RecordEvent(Profiler::EventItem::item##End, __t_ ## idx ## _ ## item ## _end_); \
    Profiler::Get()->RecordPerf(Profiler::PerfItem::item, std::chrono::duration<double, std::milli>(__t_ ## idx ## _ ## item ## _end_ - __t_ ## idx ## _ ## item ## _start_).count());

    
}

#endif