#include <common/log_as_glog_sta.h>
#include <common/util.h>
#include <common/device_manager.h>
#include <common/cuda_allocator.h>
#include <common/sm_partition.h>
#include <common/xsched_ctrl.h>
#include <common/inf_tra_comm/communicator.h>

#include <torch_col/csrc/init.h>
#include <torch_col/csrc/config.h>
#include <torch_col/csrc/torch_allocator_plugin.h>


namespace torch_col {
  
void TorchColInit(int train_rank, int train_world_size) {
  // first init config before call any other functions
  TorchColConfig::InitConfig(train_rank, train_world_size);

  COL_NVML_CALL(nvmlInit());
  colserve::sta::DeviceManager::Init();
  torch::cuda::CUDAColAllocator::CUDAColAllocator::Init();
  if (TorchColConfig::IsEnableSharedTensor()) {
    // we assume one training process one gpu 
    torch::cuda::CUDAColAllocator::CUDAColAllocator::Get()->init(train_world_size);
    torch::cuda::CUDAColAllocator::CUDAColAllocator::SetCurrentAllocator();
  }

  if (TorchColConfig::HasColocatedInferServer()) {
    colserve::ctrl::InfTraCommunicator::Init(false, false, 
                                             train_world_size);
  }

  LOG(INFO) << "TorchCol initialized.";

  CUDA_CALL(cudaSetDevice(TorchColConfig::GetTrainRank()));
  CUDA_CALL(cudaDeviceSynchronize());
}

void InitSMPartition(uint64_t stream) {
  if (!TorchColConfig::dynamic_sm_partition) {
    return ;
  }  

  colserve::SMPartitioner::Init(TorchColConfig::GetTrainRank());

  // auto stream = reinterpret_cast<cudaStream_t>(
  //   colserve::sta::xsched::GetRegisteredGlobalStream());
  // CHECK(reinterpret_cast<uint64_t>(stream) != 0);

  LOG(INFO) << "Init SMPartition, stream " << stream << " " << *(void**)stream;

  auto hook = [stream]() -> void*{
    // colserve::SetGlobalTPCMask(0x1);
    // colserve::SetStreamTpcMask(stream, 0x1);
    auto mask = colserve::SMPartitioner
        ::Get(TorchColConfig::GetTrainRank())
        ->SetTrainStreamTpcMask(reinterpret_cast<cudaStream_t>(stream));
    // LOG(INFO) << "set train stream tpc mask " << std::hex << mask;
    return nullptr;
  };

  auto succ = colserve::sta::xsched::RegisterCudaKernelLaunchPreHook(hook);
  if (!succ) {
    LOG(FATAL) << "[PySched] RegisterCudaKernelLaunchPreHook failed"; 
  }
  LOG(INFO) << "[TorchColInit] init_sm_partition done";
}


} // namespace torch_col
