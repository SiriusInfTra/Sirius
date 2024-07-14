#include <common/log_as_glog_sta.h>
#include <common/util.h>
#include <common/device_manager.h>
#include <common/cuda_allocator.h>

#include <torch_col/csrc/init.h>
#include <torch_col/csrc/config.h>
#include <torch_col/csrc/cuda_allocator_plugin.h>


namespace torch_col {
  
void TorchColInit(int train_rank, int train_world_size) {
  NVML_CALL(nvmlInit());
  TorchColConfig::InitConfig(train_rank, train_world_size);

  colserve::sta::DeviceManager::Init();
  torch::cuda::CUDAColAllocator::CUDAColAllocator::Init();
  if (TorchColConfig::IsEnableSharedTensor()) {
    // we assume one training process one gpu 
    torch::cuda::CUDAColAllocator::CUDAColAllocator::Get()->init(0);
    torch::cuda::CUDAColAllocator::CUDAColAllocator::SetCurrentAllocator();
  }
  LOG(INFO) << "TorchCol initialized.";
}

} // namespace torch_col
