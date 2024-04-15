#ifndef COLSERVE_PROFILER_H
#define COLSERVE_PROFILER_H

#include <iostream>
#include <chrono>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <fstream>
#include <unordered_map>

namespace colserve {


class Profiler {
 public:
  using time_point_t = std::chrono::time_point<std::chrono::steady_clock>;

  static std::pair<size_t, size_t> GetGPUMemInfo();
  static size_t GetLastInferMem();
  static size_t GetLastTrainMem();
  static long GetTimeStamp() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  }


  static void Init(const std::string &profile_log_path);
  static void Start();
  static void Shutdown();
  inline static time_point_t Now() {
    return std::chrono::steady_clock::now();
  }
  inline static double Milli(time_point_t begin, time_point_t end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
  }
  inline static double MilliFrom(time_point_t begin) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
  }
  static Profiler* Get();
  
  struct ResourceInfo {
    size_t infer_mem;
    size_t train_mem;
    size_t train_all_mem;
    size_t gpu_used_mem;
    size_t cold_cache_nbytes;
    double cold_cache_buffer_mb;
    double infer_mem_in_cold_cache_buffer_mb;
    double cold_cache_size_mb;
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
    InferAdjustAllocStart, // for ondemand adjust
    InferAdjustAllocEnd,
    InferLoadParamStart,
    InferLoadParamEnd,
    AddInfer,
    InferExit,
  };
  enum class PerfItem {
    // job level
    InferJobQueue,
    InferJobProcess,

    // batch level
    InferSetInput,
    InferExec,
    InferGetOutput,

    TrainAdjust,
    TrainFirstAdjust, // the latter will be batch with the first
    InferAllocStorage,
    InferWaitBeforeEnterAlloc,
    InferAdjustAlloc,
    InferLoadParam,
    InferPipelineExec,

    InferNumModelOnSwitch,

    InferRealBatchSize,

    InferModelLoad,

    InferModelColdCacheHit,

    NumPerfItem,
  };


  Profiler(const std::string &profile_log_path);
  ~Profiler();
  void RecordEvent(EventItem item);
  void RecordEvent(EventItem item, time_point_t tp);
  void RecordPerf(PerfItem item, double value);
  void RecordPerf(PerfItem item, time_point_t start, time_point_t end);

  void Clear() {
    {
      std::unique_lock lock{infer_info_mut_};
      infer_info_.clear();
    }
    {
      std::unique_lock lock{event_info_mut_};
      event_info_.clear();
    }
    {
      std::unique_lock lock{perf_info_mut_};
      perf_info_.clear();
    }
  }

  void SetWorkloadStartTimeStamp(long ts, double delay_before_profile);
  
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
  std::vector<std::tuple<double, long, ResourceInfo>> resource_info_; // pass, time stamp, resource info
  std::vector<std::tuple<double, long, EventItem>> event_info_; // pass, time stamp, event
  std::unordered_map<int, std::vector<std::tuple<long, double>>> perf_info_; // item -> [time stamp, value]
  std::mutex infer_info_mut_, event_info_mut_, perf_info_mut_;

  // pass, time stamp, infering model used memory
  std::vector<std::tuple<double, long, size_t>> infering_memory_nbytes_; 

  size_t last_infer_mem_;
  size_t last_train_mem_;

  long workload_start_time_stamp_{0};
  double delay_before_profile_{0};

  std::unique_ptr<std::thread> thread_;
};

#define PROFILE_START_WITH_ID(item, idx) \
    auto __t_ ## idx ## _ ## item ## _start_ = std::chrono::steady_clock::now(); 

#define PROFILE_START(item) \
    PROFILE_START_WITH_ID(item, 0)

// #define PROFILE_END_WITH_ID(item, idx) \
//     auto __t_ ## idx ## _ ## item ## _end_ = std::chrono::steady_clock::now(); \
//     auto __t_ ## idx ## _ ## item ## _ms_ = std::chrono::duration<double, std::milli>(__t_ ## idx ## _ ## item ## _end_ - __t_ ## idx ## _ ## item ## _start_).count(); \
//     Profiler::Get()->RecordEvent(Profiler::EventItem::item##Start, __t_ ## idx ## _ ## item ## _start_); \
//     Profiler::Get()->RecordEvent(Profiler::EventItem::item##End, __t_ ## idx ## _ ## item ## _end_); \
//     Profiler::Get()->RecordPerf(Profiler::PerfItem::item, __t_ ## idx ## _ ## item ## _ms_);

#define PROFILE_END_WITH_ID(item, idx) \
    auto __t_ ## idx ## _ ## item ## _end_ = std::chrono::steady_clock::now(); \
    auto __t_ ## idx ## _ ## item ## _ms_ = std::chrono::duration<double, std::milli>(__t_ ## idx ## _ ## item ## _end_ - __t_ ## idx ## _ ## item ## _start_).count(); \
    Profiler::Get()->RecordPerf(Profiler::PerfItem::item, __t_ ## idx ## _ ## item ## _ms_);

#define PROFILE_END(item) \
    PROFILE_END_WITH_ID(item, 0)

#define PROFILE_DURATRION_WITH_ID(item, idx) \
    __t_ ## idx ## _ ## item ## _ms_

#define PROFILE_DURATRION(item) \
    PROFILE_DURATRION_WITH_ID(item, 0)
    
}

#endif