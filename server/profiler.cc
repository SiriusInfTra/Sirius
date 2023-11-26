#include <glog/logging.h>
#include <nvml.h>
#include <regex>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "sta/cuda_allocator.h"
#include "model_train_store.h"
#include "profiler.h"
#include "config.h"

namespace colserve {
namespace {
  std::string GetMemString(size_t bytes) {
    return std::to_string(1.0 * bytes / 1024 / 1024) + " Mb";
  }
}

#define NVML_CALL(func) do{ \
    auto error = func; \
    if (error != NVML_SUCCESS) { \
      LOG(FATAL) << #func << " " << nvmlErrorString(error); \
      exit(EXIT_FAILURE); \
    } \
  } while(0);

#define CUDA_CALL(func) do { \
    auto error = func; \
    if (error != cudaSuccess) { \
    LOG(FATAL) << #func << " " << cudaGetErrorString(error); \
      exit(EXIT_FAILURE); \
    } \
  } while (0);

#define LOG_ITEM(enum, item) case enum::item: os << #item; return os;

std::ostream& operator<<(std::ostream &os, Profiler::EventItem item) {
  switch (item)
  {
    LOG_ITEM(Profiler::EventItem, TrainAdjustStart)
    LOG_ITEM(Profiler::EventItem, TrainAdjustEnd)
    LOG_ITEM(Profiler::EventItem, InferAllocStorageStart)
    LOG_ITEM(Profiler::EventItem, InferAllocStorageEnd)
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
    LOG_ITEM(Profiler::PerfItem, InferAllocStorage)
    LOG_ITEM(Profiler::PerfItem, InferLoadParam)

    LOG_ITEM(Profiler::PerfItem, InferRealBatchSize)
  default:
    return os;
  }
}

std::unique_ptr<Profiler> Profiler::profiler_;

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
  LOG(WARNING) << "USE_NVML_V3 is set to 0, profiler will not record memory info, mps server will not be checked";
#endif

  thread_.reset(new std::thread([this, device]() {
    while (!this->start_profile_) {
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    uint32_t pid = getpid();
    CUDA_CALL(cudaSetDevice(0));
    
    constexpr uint32_t max_info_cnt = 32;
    nvmlProcessInfo_t infos[32];
    while (Config::running) {
      size_t infer_mem = 0, train_mem = 0, total_mem = 0;

      if (!Config::use_shared_tensor) {
        uint32_t info_cnt = max_info_cnt;
#if !defined(USE_NVML_V3) || USE_NVML_V3 != 0
        NVML_CALL(nvmlDeviceGetComputeRunningProcesses_v3(device, &info_cnt, infos));
        CHECK(info_cnt <= 2);
        for (uint32_t i = 0; i < info_cnt; i++) {
          if (infos[i].pid == pid) {
            infer_mem = infos[i].usedGpuMemory;
          } else if (infos[i].pid == ModelTrainStore::Get()->GetTrainPid()) {
            train_mem = infos[i].usedGpuMemory;
          }
        }
#else
        infer_mem = 0;
        train_mem = 0;
#endif
        size_t free, total;
        CUDA_CALL(cudaMemGetInfo(&free, &total));
        total_mem = total - free;
      } else {
        infer_mem = sta::CUDAMemPool::InferMemUsage();
        train_mem = sta::CUDAMemPool::TrainMemUsage();
        total_mem = static_cast<size_t>(Config::cuda_memory_pool_gb * 1024 * 1024 * 1024);
      }

      this->resource_info_.push_back({this->Passed(), 
                                     {infer_mem, train_mem, total_mem}});
      // this->profile_log_ifs_ << this->Passed()
      //                        << " InferMem " << GetMemString(infer_mem)
      //                        << " TrainMem " << GetMemString(train_mem)
      //                        << std::endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
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
  event_info_.push_back({passed, item});
}

void Profiler::RecordEvent(EventItem item, Profiler::time_point_t tp) {
  auto passed = std::chrono::duration<double, std::milli>(tp - stp_).count();
  std::unique_lock lock{event_info_mut_};
  event_info_.push_back({passed, item});
}

void Profiler::RecordPerf(PerfItem item, double value) {
  auto key = static_cast<int>(item);
  std::unique_lock lock{perf_info_mut_};
  perf_info_[key].push_back(value);
}

void Profiler::RecordPerf(PerfItem item, Profiler::time_point_t start, Profiler::time_point_t end) {
  auto value = std::chrono::duration<double, std::milli>(end - start).count();
  RecordPerf(item, value);
}

double Profiler::Passed() {
  auto t = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t - stp_).count();
}

void Profiler::WriteLog() {
  std::ofstream ofs{profile_log_path_};
  
  ofs << "[Perf Info]" << std::endl;
  for (auto &it : perf_info_) {
    auto max = *std::max_element(it.second.begin(), it.second.end());
    auto min = *std::min_element(it.second.begin(), it.second.end());
    auto sum = std::accumulate(it.second.begin(), it.second.end(), 0.0);
    double avg = -1;
    if (it.second.size() > 0) {
      avg = 1.0 * sum / it.second.size();
    }
    auto sorted = it.second;
    sort(sorted.begin(), sorted.end());
    ofs << static_cast<PerfItem>(it.first) << std::fixed << std::setprecision(1) << ":"
        << " avg " << avg << " max " << max << " min " << min << " cnt " << it.second.size() << " |"
        << " p99 " << sorted[int(0.99 * sorted.size())]
        << " p95 " << sorted[int(0.95 * sorted.size())]
        << " p90 " << sorted[int(0.90 * sorted.size())]
        << " p80 " << sorted[int(0.80 * sorted.size())]
        << " p70 " << sorted[int(0.70 * sorted.size())]
        << " p60 " << sorted[int(0.60 * sorted.size())]
        << " p50 " << sorted[int(0.50 * sorted.size())]
        << std::endl;
  }
  ofs << std::endl;

  ofs << "[Event Info]" << std::endl;
  for (auto &e : event_info_) {
    ofs << std::get<0>(e) << ": "
        << std::get<1>(e) << std::endl;
  }
  ofs << std::endl;
  
  ofs << "[Memory Info]" << std::endl;
  for (auto &r : resource_info_) {
    ofs << std::get<0>(r) << ":"
        << " Infer " << GetMemString(std::get<1>(r).infer_mem)
        << " Train " << GetMemString(std::get<1>(r).train_mem)
        << " Total " << GetMemString(std::get<1>(r).gpu_used_mem)
        << std::endl;
  }
  ofs << std::endl;

  LOG(INFO) << "[Profiler] write prfile info to " << profile_log_path_;
}


} // namespace colserve
