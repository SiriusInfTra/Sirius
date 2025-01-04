#ifndef COLSERVE_PROFILER_H
#define COLSERVE_PROFILER_H

#include <common/device_manager.h>
#include <common/util.h>

#include <dcgm_agent.h>
#include <dcgm_structs.h>

#include <iostream>
#include <chrono>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <fstream>
#include <optional>
#include <unordered_map>
#include <array>


namespace colserve {

#define COL_DCGM_CALL(func) do{ \
    auto error = func; \
    if (error != DCGM_ST_OK) { \
      LOG(FATAL) << #func << " " << errorString(error); \
      exit(EXIT_FAILURE); \
    } \
  } while(0);


using time_stamp_t = double;

class Profiler {
 public:
  struct ResourceInfo;
  enum class EventItem;

  using time_point_t = std::chrono::time_point<std::chrono::steady_clock>;
  // [dcgm field id -> value] of different device
  using dcgmEntityStat = std::array<std::unordered_map<unsigned short, double>, 
                                    MAX_DEVICE_NUM>;

  using resource_entity_t = std::tuple<time_stamp_t, ResourceInfo>;
  using event_entity_t = std::tuple<time_stamp_t, EventItem>;
  using perf_entity_t = std::tuple<time_stamp_t, double>;

  static std::pair<size_t, size_t> GetGPUMemInfo();
  static size_t GetLastInferMem(int device_id);
  static size_t GetLastTrainMem();
  static size_t GetLastTrainMem(int device_id);
  static time_stamp_t GetTimeStamp() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::system_clock::now().time_since_epoch()).count();
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
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
  }
  static Profiler* Get();

  static void InferReqInc(uint64_t x = 1);
  static void InferRespInc(uint64_t x = 1);
  
  struct ResourceInfo {
    size_t infer_mem;
    size_t train_mem;
    size_t train_all_mem;
    size_t gpu_used_mem;
    size_t cold_cache_nbytes;
    double cold_cache_buffer_mb;
    double infer_mem_in_cold_cache_buffer_mb;
    double cold_cache_size_mb;

    double gpu_util;
    double gpu_mem_util;
    double sm_activity;

    int infer_required_tpc_num;
    int train_avail_tpc_num;
  };

  struct InferInfo {
    size_t id;
    double recv_time;
    double finish_time;
  };

  struct InferStat {
    uint64_t num_requests;
    uint64_t num_response;
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
    InferAdjustAlloc,
    InferLoadParam,
    InferPipelineExec,

    InferNumModelOnSwitch,

    InferRealBatchSize,

    InferModelLoad,

    InferModelColdCacheHit,

    // LLM
    LLMNumPromptTokens,
    LLMNumGenTokens,

    LLMBackendQueue, 
    LLMPrefill,
    LLMTimeBetweenTokens,

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
  void SetWorkloadEndTimeStamp(long ts);


  friend std::ostream& operator<<(std::ostream &os, EventItem item);
  friend std::ostream& operator<<(std::ostream &os, PerfItem item);

 private:
  static std::unique_ptr<Profiler> profiler_;

  void ProfileThread(std::array<nvmlDevice_t, MAX_DEVICE_NUM> devices);
  void CollectMemoryResourceInfo(
      const std::array<nvmlDevice_t, MAX_DEVICE_NUM> &devices,
      std::array<ResourceInfo, MAX_DEVICE_NUM> &res_infos /* output */);
  void CollectComputingResourcesInfo(
      const std::array<nvmlDevice_t, MAX_DEVICE_NUM> &devices, 
      std::array<ResourceInfo, MAX_DEVICE_NUM> &res_infos /* output */);

  void WriteLog();
  std::vector<std::vector<std::string>> FmtResourceInfos(
      int device_id,
      const std::vector<size_t> &field_offs,
      std::optional<time_stamp_t> start, 
      std::optional<time_stamp_t> end);
  
  template<typename field_type_t>
  std::vector<field_type_t> SelectResourceInfo(
      int device_id,
      size_t field_off, 
      std::optional<time_stamp_t> start,
      std::optional<time_stamp_t> end,
      std::optional<std::function<bool(field_type_t)>> filter = std::nullopt) {
    CHECK(device_id < sta::DeviceManager::GetNumVisibleGpu());
    std::vector<field_type_t> res;
    for (const auto &r : resource_infos_[device_id]) {
      if (start.has_value() && std::get<0>(r) < start.value()) {
        continue;
      }
      if (end.has_value() && std::get<0>(r) > end.value()) {
        continue;
      }
      auto field = reinterpret_cast<const char*>(&std::get<1>(r));
      field += field_off;
      auto value = *reinterpret_cast<const field_type_t*>(field);
      if (filter.has_value() && !filter.value()(value)) {
        continue;
      }
      res.push_back(value);
    }
    return res;
  }

  
  std::atomic<bool> start_profile_;
  std::string profile_log_path_;

  std::vector<InferInfo> infer_info_;

  // [time stamp, resource info] of different device
  std::array<std::vector<resource_entity_t>, MAX_DEVICE_NUM> resource_infos_; 

  // time stamp, event, deprecated currently
  std::vector<event_entity_t> event_info_; 

  // item -> [time stamp, value]
  std::unordered_map<int, std::vector<perf_entity_t>> perf_info_; 

  InferStat infer_stat_{0};

  std::mutex infer_info_mut_, event_info_mut_, 
             perf_info_mut_, infer_stat_mut_;

  // pass, time stamp, infering model used memory
  std::vector<std::tuple<time_stamp_t, size_t>> infering_memory_nbytes_; 

  std::array<size_t, MAX_DEVICE_NUM> last_infer_mems_;
  std::array<size_t, MAX_DEVICE_NUM> last_train_mems_;

  time_stamp_t stp_;
  time_stamp_t workload_start_time_stamp_{0};
  time_stamp_t workload_end_time_stamp_{0};
  double delay_before_profile_{0};

  dcgmHandle_t dcgm_handle_;
  dcgmGpuGrp_t dcgm_gpu_grp_;
  dcgmFieldGrp_t dcgm_field_grp_;

  int monitor_interval_ms_{100};
  std::unique_ptr<std::thread> thread_;
};

#define PROFILE_START_WITH_ID(item, idx) \
    auto __t_ ## idx ## _ ## item ## _start_ = std::chrono::steady_clock::now(); 

#define PROFILE_START(item) \
    PROFILE_START_WITH_ID(item, 0)

#define PROFILE_END_WITH_ID(item, idx) \
    auto __t_ ## idx ## _ ## item ## _end_ = std::chrono::steady_clock::now(); \
    auto __t_ ## idx ## _ ## item ## _ms_ = \
        std::chrono::duration<double, std::milli>( \
          __t_ ## idx ## _ ## item ## _end_ - __t_ ## idx ## _ ## item ## _start_).count(); \
    Profiler::Get()->RecordPerf(Profiler::PerfItem::item, __t_ ## idx ## _ ## item ## _ms_);

#define PROFILE_END(item) \
    PROFILE_END_WITH_ID(item, 0)

#define PROFILE_DURATRION_WITH_ID(item, idx) \
    __t_ ## idx ## _ ## item ## _ms_

#define PROFILE_DURATRION(item) \
    PROFILE_DURATRION_WITH_ID(item, 0)
    
}

#endif