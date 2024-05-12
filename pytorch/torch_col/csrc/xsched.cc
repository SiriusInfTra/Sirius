#include <common/log_as_glog_sta.h>
#include <common/sm_partition.h>

#include <torch_col/csrc/xsched.h>
#include <torch_col/csrc/control_stub.h>
#include <torch_col/csrc/config.h>
#include <PySched.h>

namespace torch_col {

void InitSMPartition() {
  if (!dynamic_sm_partition) {
    return ;
  }  

  colserve::SMPartitioner::Init(0, !has_colocated_infer_server);

  auto stream = reinterpret_cast<cudaStream_t>(GetRegisteredGlobalStream());
  CHECK(reinterpret_cast<uint64_t>(stream) != 0);

  auto hook = [&]() -> void*{
    // colserve::SetGlobalTPCMask(0x1);
    // colserve::SetStreamTpcMask(stream, 0x1);
    auto mask = colserve::SMPartitioner::SetTrainStreamTpcMask(stream);
    LOG(INFO) << "set train stream tpc mask " << std::hex << mask;
    return nullptr;
  };

  auto res = RegisterCudaKernelLaunchPreHook(hook);
  if (res != 0) {
    LOG(FATAL) << "[PySched] RegisterCudaKernelLaunchPreHook failed"; 
  }
  LOG(INFO) << "[PySched] init_sm_partition done";
}

}