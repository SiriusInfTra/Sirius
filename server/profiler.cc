#include <glog/logging.h>
#include <nvml.h>
#include <regex>
#include <cuda.h>
#include <cuda_runtime_api.h>

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
      LOG(FATAL) << nvmlErrorString(error); \
      exit(EXIT_FAILURE); \
    } \
  } while(0);

#define CUDA_CALL(func) do {\
    auto error = func; \
    if (error != cudaSuccess) { \
      LOG(FATAL) << cudaGetErrorString(error); \
      exit(EXIT_FAILURE); \
    } \
  } while (0);

std::ostream& operator<<(std::ostream &os, Profiler::EventItem item) {
#define LOG_EventItem(enum, item) \
  case enum::item: os << #item; return os;

  switch (item)
  {
    LOG_EventItem(Profiler::EventItem, TrainAdjustStart)
    LOG_EventItem(Profiler::EventItem, TrainAdjustEnd)
    LOG_EventItem(Profiler::EventItem, InferAllocStorageStart)
    LOG_EventItem(Profiler::EventItem, InferAllocStorageEnd)
    LOG_EventItem(Profiler::EventItem, InferLoadParamStart)
    LOG_EventItem(Profiler::EventItem, InferLoadParamEnd)
    LOG_EventItem(Profiler::EventItem, AddInfer)
    LOG_EventItem(Profiler::EventItem, InferExit)
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

  thread_.reset(new std::thread([this]() {
    while (!this->start_profile_) {
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    uint32_t pid = getpid();
    uint32_t dev_cnt;
    NVML_CALL(nvmlDeviceGetCount_v2(&dev_cnt));
    CHECK_GT(dev_cnt, 0);

    uint32_t dev_id = 0;
    nvmlDevice_t device;
    auto visiable_device = std::getenv("CUDA_VISIBLE_DEVICES");
    if (visiable_device != nullptr) {
      auto s = std::string(visiable_device);
      std::regex r{"[0-9]+"};
      std::smatch m;
      if (std::regex_search(s, m, r)) {
        dev_id = std::stoi(m.str());
      }
    }
    
    NVML_CALL(nvmlDeviceGetHandleByIndex(dev_id, &device));
    CUDA_CALL(cudaSetDevice(0));
    
    constexpr uint32_t max_info_cnt = 32;
    uint32_t info_cnt;
    nvmlProcessInfo_t infos[32];
    while (Config::running) {
      info_cnt = max_info_cnt;
      NVML_CALL(nvmlDeviceGetComputeRunningProcesses_v3(device, &info_cnt, infos));
      // CHECK(info_cnt <= 2);
      
      size_t infer_mem = 0, train_mem = 0;
      for (uint32_t i = 0; i < info_cnt; i++) {
        if (infos[i].pid == pid) {
          infer_mem = infos[i].usedGpuMemory;
        } else if (infos[i].pid == ModelTrainStore::Get()->GetTrainPid()) {
          train_mem = infos[i].usedGpuMemory;
        }
      }

      size_t free, total;
      CUDA_CALL(cudaMemGetInfo(&free, &total));
      this->resource_info_.push_back({this->Passed(), 
                                     {infer_mem, train_mem, total - free}});
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
  std::unique_lock lock{event_info_mut_};
  event_info_.push_back({Passed(), item});
}

double Profiler::Passed() {
  auto t = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t - stp_).count();
}

void Profiler::WriteLog() {
  std::ofstream ofs{profile_log_path_};
  ofs << "[Memory Info]" << std::endl;
  for (auto &r : resource_info_) {
    ofs << std::get<0>(r) << ":"
        << " Infer " << GetMemString(std::get<1>(r).infer_mem)
        << " Train " << GetMemString(std::get<1>(r).train_mem)
        << " Total " << GetMemString(std::get<1>(r).gpu_used_mem)
        << std::endl;
  }
  ofs << std::endl;
  ofs << "[Event Info]" << std::endl;
  for (auto &e : event_info_) {
    ofs << std::get<0>(e) << ": "
        << std::get<1>(e) << std::endl;
  }
  LOG(INFO) << "[Profiler] write prfile info to " << profile_log_path_;
}


} // namespace colserve
