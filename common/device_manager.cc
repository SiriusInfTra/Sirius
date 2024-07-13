#include "log_as_glog_sta.h"
#include <common/device_manager.h>
#include <common/util.h>

#include <boost/algorithm/string.hpp>
#include <regex>

namespace colserve::sta {

std::unique_ptr<DeviceManager> DeviceManager::device_manager_ = nullptr;

DeviceManager::DeviceManager() {
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

void DeviceManager::Init() {
  if (device_manager_ == nullptr) {
    device_manager_ = std::make_unique<DeviceManager>();
  }
}

int DeviceManager::GetNumGpu() {
  CHECK(device_manager_ != nullptr);
  return device_manager_->system_gpu_uuids_.size();
}

int DeviceManager::GetNumVisibleGpu() {
  CHECK(device_manager_ != nullptr);
  return device_manager_->gpu_id_map_.size();
}

int DeviceManager::GetGpuSystemId(int gpu_id) {
  CHECK(device_manager_ != nullptr);
  auto iter = device_manager_->gpu_id_map_.find(gpu_id);
  CHECK(iter != device_manager_->gpu_id_map_.end());
  return iter->second.first;
}

const std::string& DeviceManager::GetGpuSystemUuid(int gpu_id) {
  CHECK(device_manager_ != nullptr);
  auto iter = device_manager_->gpu_id_map_.find(gpu_id);
  CHECK(iter != device_manager_->gpu_id_map_.end());
  return iter->second.second;
}

} // namespace colserve::sta