#include <common/log_as_glog_sta.h>
#include <common/xsched_ctrl.h>

#include <xsched/include/shim/cuda/xctrl.h>
#include <xsched/include/shim/cuda/shim.h>
#include <xsched/include/shim/common/xmanager.h>
#include <xsched/include/hal/cuda/cuda_command.h>
#include <xsched/include/preempt/xqueue/xcommand.h>
#include <xsched/include/preempt/xqueue/xqueue.h>

#include <memory>
#include <atomic>

namespace colserve {
namespace sta {
namespace xsched {

static std::atomic<uint64_t> global_stream_handle;
static std::atomic<uint64_t> global_stream;

uint64_t RegisterStream(uint64_t stream) {
  auto cuda_stream = reinterpret_cast<CUstream>(stream);
  global_stream = stream;
  global_stream_handle = CudaXQueueCreate(cuda_stream, 
      PREEMPT_MODE_STOP_SUBMISSION, 16, 8);
  LOG(INFO) << "Xsched register stream " << stream 
            << " with global handle: " << global_stream_handle;
  return global_stream_handle;
}

uint64_t GetRegisteredGlobalHandle() {
  return global_stream_handle;
}

uint64_t GetRegisteredGlobalStream() {
  return global_stream;
}

void UnRegisterStream() {
  CudaXQueueDestroy(global_stream_handle);
}

size_t AbortStream() {
  using namespace ::xsched::hal::cuda;
  using namespace ::xsched::preempt;
  auto xqueue = CudaXQueueGet(global_stream_handle);
  CHECK(xqueue != nullptr) << "Required xqueue is not NULL, global_stream_handle=" 
                           << global_stream_handle;
  auto remove_filter = [](std::shared_ptr<XCommand> hal_command) -> bool {
    return std::dynamic_pointer_cast<CudaMemoryCommand>(hal_command) != nullptr 
        || std::dynamic_pointer_cast<CudaKernelLaunchCommand>(hal_command) != nullptr;
  };
  return xqueue->Clear(remove_filter);
}

bool SyncStream() {
  return CudaXQueueSync(global_stream_handle);
}

size_t GetXQueueSize() {
  auto xqueue = CudaXQueueGet(global_stream_handle);
  CHECK(xqueue != nullptr) << "Required xqueue is not NULL, global_stream_handle=" 
                           << global_stream_handle;
  return xqueue->GetSize();
}

bool QueryRejectCudaCalls() {
  return CudaXQueueQueryReject();
}

void SetRejectCudaCalls(bool reject) {
  CudaXQueueSetReject(reject);
}

bool RegisterCudaKernelLaunchPreHook(std::function<void*()> fn) {
  using CudaKernelLaunchCommand = ::xsched::hal::cuda::CudaKernelLaunchCommand;
  return CudaKernelLaunchCommand::RegisterCudaKernelLaunchPreHook(fn);
}


} // namesapce xsched
} // namespace sta
} // namespace colserve