#ifndef COLSERVE_DEVICE_MANAGER_H
#define COLSERVE_DEVICE_MANAGER_H

#include <iostream>
#include <vector>
#include <unordered_map>
#include <memory>

namespace colserve::sta {

#define STA_CURRENT_DEVICE sta::DeviceManager::GetCurrentDevice()

class DeviceGuard {
 public:
  DeviceGuard(int device_id);
  ~DeviceGuard();
  
 private:
  int prev_device_id_;
  int device_id_;
};


class DeviceManager {
 public:
  DeviceManager();
  
  static void Init();
  static int GetNumGpu();
  static int GetNumVisibleGpu();
  static int GetGpuSystemId(int gpu_id);
  static const std::string& GetGpuSystemUuid(int gpu_id);
  static int GetCurrentDevice();
  static int GetVisibleGpuId(int system_id);
  
 private:
  static std::unique_ptr<DeviceManager> device_manager_;

  // uuid of all system gpu
  std::vector<std::string> system_gpu_uuids_;

  // visible id -> (system id, uuid)
  std::unordered_map<int, std::pair<int, std::string>> gpu_id_map_;
  std::unordered_map<int, int> system_to_visible_id_;

};

using DevMgr = DeviceManager;

} // namespace colserve::sta

#endif