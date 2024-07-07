#include "logging_as_glog.h"
#include <common/cuda_allocator.h>
#include <common/util.h>

#include <server/resource_manager.h>
#include <server/train_launcher.h>
#include <server/config.h>

#include <regex>
#include <boost/algorithm/string.hpp>


namespace colserve {

std::unique_ptr<ResourceManager> ResourceManager::resource_manager_;

ResourceManager::ResourceManager() {
  auto visible_gpu_env = std::getenv("CUDA_VISIBLE_DEVICES");
  uint32_t num_gpus;
  NVML_CALL(nvmlDeviceGetCount(&num_gpus));
  for (int i = 0; i < num_gpus; i++) {
    nvmlDevice_t device;
    NVML_CALL(nvmlDeviceGetHandleByIndex(i, &device));
    char uuid[128];
    NVML_CALL(nvmlDeviceGetUUID(device, uuid, 128));
    system_gpu_uuids_.push_back(uuid);
  }
  if (visible_gpu_env) {
    std::string visible_gpu_str(visible_gpu_env);
    // std::vector<std::string> visible_gpu_strs = 
    std::vector<std::string> visible_gpu_strs;
    boost::algorithm::split(visible_gpu_strs, visible_gpu_str, boost::is_any_of(","));
    for (int i = 0; i < visible_gpu_strs.size(); i++) {
      try {
        int gpu_id = std::stoi(visible_gpu_strs[i]);
        if (gpu_id < 0 || gpu_id >= num_gpus) {
          LOG(FATAL) << "Invalid CUDA_VISIBLE_DEVICES " << visible_gpu_strs[i];
        }
        gpu_id_map_[i] = {gpu_id, system_gpu_uuids_[gpu_id]};
      } catch (std::invalid_argument &e) {
        std::regex r{"GPU-[^ ,]+"};
        std::smatch m;
        if (std::regex_search(visible_gpu_strs[i], m, r)) {
          std::string gpu_uuid = m.str();
          auto it = std::find(system_gpu_uuids_.begin(), system_gpu_uuids_.end(), gpu_uuid);
          if (it == system_gpu_uuids_.end()) {
            LOG(FATAL) << "Invalid CUDA_VISIBLE_DEVICES " << visible_gpu_strs[i];
          }
          auto gpu_sys_id = it - system_gpu_uuids_.begin();
          gpu_id_map_[i] = {gpu_sys_id, system_gpu_uuids_[gpu_sys_id]};
        } else {
          LOG(FATAL) << "Invalid CUDA_VISIBLE_DEVICES " << visible_gpu_strs[i];
        }
      } 
    }
  } else {
    for (int i = 0; i < num_gpus; i++) {
      gpu_id_map_[i] = {i, system_gpu_uuids_[i]};
    }
  }
}

double ResourceManager::GetFreeMemoryMB(bool verbose) {
  using namespace sta;

  double free_memory_mb;
  double infer_memory_mb = GetInferMemoryMB();
  double train_memory_mb = GetTrainMemoryMB();
  double train_predict_memory_mb = TrainLauncher::Get()->PredictMemUsageMB(verbose);

  if (Config::use_shared_tensor) {
    free_memory_mb = sta::ByteToMB(sta::CUDAMemPool::PoolNbytes());
    free_memory_mb -= infer_memory_mb;
    free_memory_mb -= std::max(train_memory_mb, train_predict_memory_mb);
    free_memory_mb -= Config::train_memory_over_predict_mb;
  } else {
    auto [free, total] = Profiler::GetGPUMemInfo();
    free_memory_mb = ByteToMB(free);
    free_memory_mb = std::min(
        free_memory_mb, 
        sta::ByteToMB(total) 
        - infer_memory_mb 
        - std::max(train_predict_memory_mb, train_memory_mb) 
        - Config::train_memory_over_predict_mb
    );
  }

  LOG_IF(INFO, verbose && Config::log_memory_adjust) 
      << "[ResourceManager] "
      << " infer memory " << infer_memory_mb 
      << " train memory " << train_memory_mb 
      << " predict train memory " << train_predict_memory_mb
      << " free memory " << free_memory_mb;
            
  return free_memory_mb;
}

double ResourceManager::GetTrainAvailMemoryMB(bool verbose) {
  using namespace sta;

  double infer_memory_mb = GetInferMemoryMB();

  double free_memory_mb;
  if (Config::use_shared_tensor) {
    free_memory_mb = sta::ByteToMB(sta::CUDAMemPool::PoolNbytes());
    free_memory_mb -= infer_memory_mb;
    free_memory_mb -= Config::train_memory_over_predict_mb;
  } else {
    auto [free, total] = Profiler::GetGPUMemInfo();
    free_memory_mb = ByteToMB(free);
    free_memory_mb = std::min(
        free_memory_mb, 
        sta::ByteToMB(total) - infer_memory_mb - Config::train_memory_over_predict_mb
    );
  }

    LOG_IF(INFO, verbose) << "[ResourceManager]"
                          << " free memory " << free_memory_mb
                          << " infer memory " << infer_memory_mb;

  return free_memory_mb;  
}

void ResourceManager::InferMemoryChangingLock() {
  CHECK(resource_manager_ != nullptr);
  resource_manager_->infer_memory_changing_mut_.lock();
}

void ResourceManager::InferMemoryChangingUnlock() {
  CHECK(resource_manager_ != nullptr);
  resource_manager_->infer_memory_changing_mut_.unlock();
}

bool ResourceManager::InferChangeMemoryTryLock() {
  CHECK(resource_manager_ != nullptr);
  return resource_manager_->infer_memory_changing_mut_.try_lock();
}

double ResourceManager::GetInferMemoryMB() {
  using namespace sta;
  if (Config::use_shared_tensor_infer) {
    return ByteToMB(CUDAMemPool::InferMemUsage());
  } else {
    return ByteToMB(Profiler::GetLastInferMem());
  }
}

double ResourceManager::GetTrainMemoryMB() {
  using namespace sta;
  if (Config::use_shared_tensor_train) {
    return ByteToMB(CUDAMemPool::TrainAllMemUsage());
  } else {
    return ByteToMB(Profiler::GetLastTrainMem());
  }
}

int ResourceManager::GetNumGpu() {
  CHECK(resource_manager_ != nullptr);
  return resource_manager_->system_gpu_uuids_.size();
}

int ResourceManager::GetNumVisibleGpu() {
  CHECK(resource_manager_ != nullptr);
  return resource_manager_->gpu_id_map_.size();
}

int ResourceManager::GetGpuSystemId(int gpu_id) {
  CHECK(resource_manager_ != nullptr);
  auto iter = resource_manager_->gpu_id_map_.find(gpu_id);
  CHECK(iter != resource_manager_->gpu_id_map_.end());
  return iter->second.first;
}

const std::string& ResourceManager::GetGpuSystemUuid(int gpu_id) {
  CHECK(resource_manager_ != nullptr);
  auto iter = resource_manager_->gpu_id_map_.find(gpu_id);
  CHECK(iter != resource_manager_->gpu_id_map_.end());
  return iter->second.second;
}


}