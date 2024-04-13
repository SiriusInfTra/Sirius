#include "logging_as_glog.h"
#include <common/cuda_allocator.h>
#include <common/util.h>
#include <server/infer_model_store.h>

#include "train_launcher.h"
#include "profiler.h"
#include "config.h"

#include <nvml.h>
#include <numeric>
#include <regex>

namespace colserve {
namespace {
  std::string GetMemString(size_t bytes) {
    return std::to_string(1.0 * bytes / 1024 / 1024) + " Mb";
  }
}

#define LOG_ITEM(enum, item) case enum::item: os << #item; return os;

std::ostream& operator<<(std::ostream &os, Profiler::EventItem item) {
  switch (item)
  {
    LOG_ITEM(Profiler::EventItem, TrainAdjustStart)
    LOG_ITEM(Profiler::EventItem, TrainAdjustEnd)
    LOG_ITEM(Profiler::EventItem, InferAllocStorageStart)
    LOG_ITEM(Profiler::EventItem, InferAllocStorageEnd)
    LOG_ITEM(Profiler::EventItem, InferAdjustAllocStart)
    LOG_ITEM(Profiler::EventItem, InferAdjustAllocEnd)
    LOG_ITEM(Profiler::EventItem, InferLoadParamStart)
    LOG_ITEM(Profiler::EventItem, InferLoadParamEnd)
    LOG_ITEM(Profiler::EventItem, AddInfer)
    LOG_ITEM(Profiler::EventItem, InferExit)
  default:
    return os;
  }
}

std::ostream& operator<<(std::ostream &os, Profiler::PerfItem item) {
  switch (item) {
    LOG_ITEM(Profiler::PerfItem, InferJobQueue)
    LOG_ITEM(Profiler::PerfItem, InferJobProcess)

    LOG_ITEM(Profiler::PerfItem, InferSetInput)
    LOG_ITEM(Profiler::PerfItem, InferExec)
    LOG_ITEM(Profiler::PerfItem, InferGetOutput)

    LOG_ITEM(Profiler::PerfItem, TrainAdjust)
    LOG_ITEM(Profiler::PerfItem, TrainFirstAdjust)
    LOG_ITEM(Profiler::PerfItem, InferAllocStorage)
    LOG_ITEM(Profiler::PerfItem, InferWaitBeforeEnterAlloc)
    LOG_ITEM(Profiler::PerfItem, InferAdjustAlloc)
    LOG_ITEM(Profiler::PerfItem, InferLoadParam)
    LOG_ITEM(Profiler::PerfItem, InferPipelineExec)

    LOG_ITEM(Profiler::PerfItem, InferNumModelOnSwitch)

    LOG_ITEM(Profiler::PerfItem, InferRealBatchSize)
    LOG_ITEM(Profiler::PerfItem, InferModelLoad)
    LOG_ITEM(Profiler::PerfItem, InferModelColdCacheHit)
  default:
    return os;
  }
}

std::unique_ptr<Profiler> Profiler::profiler_;

std::pair<size_t, size_t> Profiler::GetGPUMemInfo() {
  size_t free, total;
  CUDA_CALL(cudaMemGetInfo(&free, &total));
  return {free, total};
}

size_t Profiler::GetLastInferMem() {
  CHECK(profiler_ != nullptr);
  return profiler_->last_infer_mem_;
}

size_t Profiler::GetLastTrainMem() {
  CHECK(profiler_ != nullptr);
  return profiler_->last_train_mem_;
}

void Profiler::Init(const std::string &profile_log_path) {
  CHECK(profiler_ == nullptr);
  profiler_ = std::make_unique<Profiler>(profile_log_path);
}

void Profiler::Start() {
  std::unique_lock infer_info_lock{profiler_->infer_info_mut_};
  // std::unique_lock event_info_lock{profiler_->event_info_mut_};
  profiler_->infer_info_.clear();
  // profiler_->event_info_.clear();
  profiler_->resource_info_.clear();
  profiler_->infering_memory_nbytes_.clear();
  profiler_->start_profile_ = true;
}

void Profiler::Shutdown() {
  profiler_->thread_->join();
  NVML_CALL(nvmlShutdown());
  profiler_->WriteLog();
}

Profiler* Profiler::Get() {
  CHECK(profiler_ != nullptr);
  return profiler_.get();
}

Profiler::Profiler(const std::string &profile_log_path)
    : profile_log_path_(profile_log_path), start_profile_(false) {
  NVML_CALL(nvmlInit());
  stp_ = std::chrono::steady_clock::now();

  uint32_t dev_cnt;
  NVML_CALL(nvmlDeviceGetCount_v2(&dev_cnt));
  CHECK_GT(dev_cnt, 0);

  // uint32_t dev_id = 0;
  std::string gpu_uuid;
  nvmlDevice_t device;
  auto visiable_device = std::getenv("CUDA_VISIBLE_DEVICES");
  if (visiable_device != nullptr) {
    auto s = std::string(visiable_device);
    std::regex r{"GPU-[^ ,]+"};
    std::smatch m;
    if (std::regex_search(s, m, r)) {
      // dev_id = std::stoi(m.str());
      gpu_uuid = m.str();
    } else {
      LOG(FATAL) << "please use UUID in CUDA_VISIBLE_DEVICES, "
                 << "found $CUDA_VISIBLE_DEVICES="<< visiable_device << ", "
                 << "get UUID by nvidia-smi -L";
    }
  }

  if (!gpu_uuid.empty()) {
    NVML_CALL(nvmlDeviceGetHandleByUUID(gpu_uuid.c_str(), &device));
  } else {
    NVML_CALL(nvmlDeviceGetHandleByIndex(0, &device));
  }

  // CHECK MPS
  if (colserve::Config::check_mps) {
    uint32_t info_cnt = 0;
#if !defined(USE_NVML_V3) || USE_NVML_V3 != 0
    auto nvml_err = nvmlDeviceGetMPSComputeRunningProcesses_v3(device, &info_cnt, NULL);
    if (nvml_err == NVML_SUCCESS && info_cnt == 0) {
      LOG(FATAL) << "MPS is not enabled, please start MPS server by nvidia-cuda-mps-control";
    }
#endif
  }

#if defined(USE_NVML_V3) && USE_NVML_V3 == 0
  LOG(FATAL) << "USE_NVML_V3 is set to 0, profiler will not record memory info, mps server will not be checked";
#endif

  thread_.reset(new std::thread([this, device]() {
    while (!this->start_profile_) {
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    LOG(INFO) << "start profiler thread";
    uint32_t pid = getpid();
    CUDA_CALL(cudaSetDevice(0));
    
    constexpr uint32_t max_info_cnt = 32;
    nvmlProcessInfo_t infos[32];
    while (Config::running) {
      size_t infer_mem = 0, train_mem = 0, train_all_mem = 0, total_mem = 0;
      size_t cold_cache_nbytes = 0;
      double cold_cache_buffer_mb = 0;
      double infer_mem_in_cold_cache_buffer_mb = 0;
      double cold_cache_size_mb = 0;
      if (!Config::use_shared_tensor || !Config::use_shared_tensor_train) {
        uint32_t info_cnt = max_info_cnt;
        NVML_CALL(nvmlDeviceGetComputeRunningProcesses_v3(device, &info_cnt, infos));
        for (uint32_t i = 0; i < info_cnt; i++) {
          if (!Config::use_shared_tensor_train && infos[i].pid == pid) {
            infer_mem = infos[i].usedGpuMemory;
          } else if (infos[i].pid == TrainLauncher::Get()->GetTrainPid()) {
            train_mem = infos[i].usedGpuMemory;
            train_all_mem = train_mem;
          }
        }
        size_t free, total;
        CUDA_CALL(cudaMemGetInfo(&free, &total));
        total_mem = total - free;
        if (Config::use_shared_tensor) {
          infer_mem = sta::CUDAMemPool::InferMemUsage();
        }
      } else {
        infer_mem = sta::CUDAMemPool::InferMemUsage();
        train_mem = sta::CUDAMemPool::TrainMemUsage();
        train_all_mem = sta::CUDAMemPool::TrainAllMemUsage();
        total_mem = static_cast<size_t>(Config::cuda_memory_pool_gb * 1_GB);
        if (Config::cold_cache_max_capability_nbytes != 0) {
          cold_cache_nbytes = ColdModelCache::Get().GetCachedNbytesUnsafe();
          cold_cache_buffer_mb = ColdModelCache::Get().GetBufferMBUnsafe();
          infer_mem_in_cold_cache_buffer_mb = ColdModelCache::Get().GetColdCacheReleasableMemoryMBUnsafe();
          cold_cache_size_mb = ColdModelCache::Get().GetCacheSizeMBUnsafe();
        }
      }
      this->last_infer_mem_ = infer_mem;
      this->last_train_mem_ = train_mem;
      this->resource_info_.push_back({this->Passed(), Profiler::GetTimeStamp(),
          ResourceInfo{.infer_mem = infer_mem, .train_mem = train_mem, 
                       .train_all_mem = train_all_mem, .gpu_used_mem = total_mem, 
                       .cold_cache_nbytes = cold_cache_nbytes, 
                       .cold_cache_buffer_mb = cold_cache_buffer_mb,
                       .infer_mem_in_cold_cache_buffer_mb = infer_mem_in_cold_cache_buffer_mb,
                       .cold_cache_size_mb = cold_cache_size_mb}});
      this->infering_memory_nbytes_.push_back({this->Passed(), Profiler::GetTimeStamp(),
                                              InferModelStore::GetInferingModelNbytes()});
      // this->profile_log_ifs_ << this->Passed()
      //                        << " InferMem " << GetMemString(infer_mem)
      //                        << " TrainMem " << GetMemString(train_mem)
      //                        << std::endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }));
}

Profiler::~Profiler() {
  // NVML_CALL(nvmlShutdown());
  // profile_log_ifs_.close();
}

void Profiler::RecordEvent(EventItem item) {
  auto passed = Passed();
  std::unique_lock lock{event_info_mut_};
  event_info_.push_back({passed, Profiler::GetTimeStamp(), item});
}

void Profiler::RecordEvent(EventItem item, Profiler::time_point_t tp) {
  auto passed = std::chrono::duration<double, std::milli>(tp - stp_).count();
  std::unique_lock lock{event_info_mut_};
  event_info_.push_back({passed, Profiler::GetTimeStamp(), item});
}

void Profiler::RecordPerf(PerfItem item, double value) {
  auto key = static_cast<int>(item);
  std::unique_lock lock{perf_info_mut_};
  perf_info_[key].push_back({Profiler::GetTimeStamp(), value});
}

void Profiler::RecordPerf(PerfItem item, Profiler::time_point_t start, Profiler::time_point_t end) {
  auto value = std::chrono::duration<double, std::milli>(end - start).count();
  RecordPerf(item, value);
}

void Profiler::SetWorkloadStartTimeStamp(long ts, double delay_before_profile) {
  workload_start_time_stamp_ = ts;
  delay_before_profile_ = delay_before_profile;
  LOG(INFO) << "workload start at " << ts 
            << ", profile will start after " << delay_before_profile
            << " sec from this time point";
}

double Profiler::Passed() {
  auto t = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t - stp_).count();
}

void Profiler::WriteLog() {
  std::ofstream ofs{profile_log_path_};
  
  ofs << "[Perf Info] workload start time stamp " << workload_start_time_stamp_ 
      << " delay before profile " << delay_before_profile_ << " sec " << std::endl;
  for (auto &it : perf_info_) {
    auto item = static_cast<Profiler::PerfItem>(it.first);
    std::vector<double> item_perf_info;
    for (auto &p : it.second) {
      auto time_stamp = std::get<0>(p);
      auto value = std::get<1>(p);
      if (time_stamp > workload_start_time_stamp_ + static_cast<long>(delay_before_profile_ * 1000)) {
        item_perf_info.push_back(value);
      }
    }
    if (item_perf_info.empty()) {
      ofs << item << ": no record after workload start profile time stamp" << std::endl;
      continue;
    }
    auto max = *std::max_element(item_perf_info.begin(), item_perf_info.end());
    auto min = *std::min_element(item_perf_info.begin(), item_perf_info.end());
    auto sum = std::accumulate(item_perf_info.begin(), item_perf_info.end(), 0.0);
    double avg = -1;
    if (item_perf_info.size() > 0) {
      avg = 1.0 * sum / item_perf_info.size();
    }
    auto sorted = item_perf_info;
    sort(sorted.begin(), sorted.end());
    ofs << static_cast<PerfItem>(it.first) << std::fixed << std::setprecision(1) << ":"
        << " avg " << avg << " max " << max << " min " << min << " cnt " << item_perf_info.size() << " |"
        << " p99 " << sorted[int(0.99 * sorted.size())]
        << " p95 " << sorted[int(0.95 * sorted.size())]
        << " p90 " << sorted[int(0.90 * sorted.size())]
        << " p80 " << sorted[int(0.80 * sorted.size())]
        << " p70 " << sorted[int(0.70 * sorted.size())]
        << " p60 " << sorted[int(0.60 * sorted.size())]
        << " p50 " << sorted[int(0.50 * sorted.size())]
        << std::endl;
  }
  for (int i = 0; i < static_cast<int>(Profiler::PerfItem::NumPerfItem); i++) {
    auto item = static_cast<Profiler::PerfItem>(i);
    if (perf_info_.find(i) == perf_info_.end()) {
      ofs << item << ": no record" << std::endl;
    }
  }
  ofs << std::endl;

  ofs << "[Event Info]" << std::endl;
  for (auto &e : event_info_) {
    ofs << std::get<0>(e) << ": "
        << std::get<1>(e) << ": "
        << std::get<2>(e) << std::endl;
  }
  ofs << std::endl;
  
  ofs << "[Memory Info]" << std::endl;
  for (auto &r : resource_info_) {
    ofs << std::get<0>(r) << ": " 
        << std::get<1>(r) << ":"
        << " Infer " << GetMemString(std::get<2>(r).infer_mem)
        << " Train " << GetMemString(std::get<2>(r).train_mem)
        << " TrainAll " << GetMemString(std::get<2>(r).train_all_mem)
        << " Total " << GetMemString(std::get<2>(r).gpu_used_mem)
        << " | ColdCache " << GetMemString(std::get<2>(r).cold_cache_nbytes)
        << " ColdCacheBuffer " << std::get<2>(r).cold_cache_buffer_mb << " Mb"
        << " InferInColdCacheBuffer " << std::get<2>(r).infer_mem_in_cold_cache_buffer_mb << " Mb"
        << " ColdCacheSize " << std::get<2>(r).cold_cache_size_mb << " Mb"
        << std::endl;
  }
  ofs << std::endl;

  // ofs << "[Infering Model Memory Info]" << std::endl;
  // for (auto &x : infering_memory_nbytes_) {
  //   ofs << std::get<0>(x) << ": "
  //       << std::get<1>(x) << ": "
  //       << std::get<2>(x) << std::endl;
  // }

  ofs << std::endl;

  // int hit_count = std::accumulate()
  // ofs << "[Cache Info]" << "Hit: " << << std::endl;

  LOG(INFO) << "[Profiler] write prfile info to " << profile_log_path_;
}


} // namespace colserve
