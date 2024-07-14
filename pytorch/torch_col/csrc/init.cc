#include <common/log_as_glog_sta.h>
#include <common/util.h>
#include <common/device_manager.h>
#include <common/cuda_allocator.h>

#include <torch_col/csrc/init.h>
#include <torch_col/csrc/config.h>
#include <torch_col/csrc/cuda_allocator_plugin.h>


namespace torch_col {
  
void TorchColInit(int device_count) {
  NVML_CALL(nvmlInit());
  TorchColConfig::InitConfig();
  colserve::sta::DeviceManager::Init();
  CHECK(colserve::sta::DeviceManager::GetNumVisibleGpu() == device_count);
  torch::cuda::CUDAColAllocator::CUDAColAllocator::Init();
  if (TorchColConfig::IsEnableSharedTensor()) {
    torch::cuda::CUDAColAllocator::CUDAColAllocator::Get()->init(device_count);
    torch::cuda::CUDAColAllocator::CUDAColAllocator::SetCurrentAllocator();
  }
  LOG(INFO) << "TorchCol initialized.";
}

} // namespace torch_col
