#include <common/log_as_glog_sta.h>
#include <common/xsched_ctrl.h>

#include <xsched/include/shim/cuda/xctrl.h>
#include <xsched/include/shim/cuda/shim.h>
#include <xsched/include/shim/cuda/extra.h>
#include <xsched/include/shim/common/xmanager.h>
#include <xsched/include/hal/cuda/cuda_command.h>
#include <xsched/include/preempt/xqueue/xcommand.h>
#include <xsched/include/preempt/xqueue/xqueue.h>

#include <memory>
#include <atomic>
#include <algorithm>
#include <numeric>

namespace colserve {
namespace sta {
namespace xsched {

// static std::atomic<uint64_t> global_stream_handle;
// static std::atomic<uint64_t> global_stream;

static std::mutex stream_mut;
static std::unordered_map<uint64_t, uint64_t> stream_map;

uint64_t RegisterStream(uint64_t stream) {
  auto cuda_stream = reinterpret_cast<CUstream>(stream);

  std::unique_lock lock{stream_mut};
  auto it = stream_map.find(stream);
  if (it != stream_map.end()) {
    LOG(FATAL) << "Stream 0x" << stream 
               << " already registered with handle 0x" << it->second;
  }

  stream_map[stream] = CudaXQueueCreate(cuda_stream, 
      PREEMPT_MODE_STOP_SUBMISSION, 16, 8);
  // global_stream = stream;
  // global_stream_handle = CudaXQueueCreate(cuda_stream, 
  //     PREEMPT_MODE_STOP_SUBMISSION, 16, 8);
  LOG(INFO) << std::hex
            << "Xsched register stream 0x" << stream 
            << " with global handle: 0x" << stream_map[stream];
  return stream_map[stream];
}

uint64_t GetRegisteredHandleWithLock(uint64_t stream, 
                             std::unique_lock<std::mutex> &lock) {
  auto it = stream_map.find(stream);
  if (it == stream_map.end()) {
    LOG(FATAL) << "Stream 0x" << stream << " not registered";
  }
  return it->second;
}

uint64_t GetRegisteredHandle(uint64_t stream) {
  std::unique_lock lock{stream_mut};
  return GetRegisteredHandleWithLock(stream, lock);
}

std::vector<uint64_t> GetRegisteredStreams() {
  return std::accumulate(stream_map.begin(), stream_map.end(), 
      std::vector<uint64_t>{}, 
      [](std::vector<uint64_t> &acc, const std::pair<uint64_t, uint64_t> &kv) {
        acc.push_back(kv.first);
        return acc;
      });
}

void UnRegisterStream(uint64_t stream) {
  std::unique_lock lock{stream_mut};
  auto it = stream_map.find(stream);
  if (it == stream_map.end()) {
    LOG(FATAL) << "Stream 0x" << stream << " not registered";
  }
  CudaXQueueDestroy(stream_map[stream]);
  stream_map.erase(stream);
}

void UnRegisterAllStreams() {
  std::unique_lock lock{stream_mut};
  for (auto &kv : stream_map) {
    CudaXQueueDestroy(kv.second);
  }
  stream_map.clear();
}

size_t AbortStreamHandleWithLock(uint64_t stream_handle, 
                                 std::unique_lock<std::mutex> &lock) {
  using namespace ::xsched::hal::cuda;
  using namespace ::xsched::preempt;
  auto xqueue = CudaXQueueGet(stream_handle);
  CHECK(xqueue != nullptr) << "XQueue is NULL, stream_handle=" 
                           << stream_handle;
  auto remove_filter = [](std::shared_ptr<XCommand> hal_command) -> bool {
    // if (auto kernel = std::dynamic_pointer_cast<CudaKernelLaunchCommand>(hal_command);
    //     kernel != nullptr && 
    //     std::string(::xsched::shim::cuda::extra::GetFuncName(kernel->function_))
    //         .find("ncclKernel") != std::string::npos) {
    //     LOG(INFO) << "Abort nccl kernel: " << kernel.get() 
    //               << " " << ::xsched::shim::cuda::extra::GetFuncName(kernel->function_);
    //   }
    return std::dynamic_pointer_cast<CudaMemoryCommand>(hal_command) != nullptr 
        || std::dynamic_pointer_cast<CudaKernelLaunchCommand>(hal_command) != nullptr;
  };
  return xqueue->Clear(remove_filter);
}

size_t AbortStream(uint64_t stream) {
  std::unique_lock lock{stream_mut};
  auto it = stream_map.find(stream);
  auto stream_handle = GetRegisteredHandleWithLock(stream, lock);
  return AbortStreamHandleWithLock(stream_handle, lock);
}

size_t AbortAllStreams() {
  std::unique_lock lock{stream_mut};
  size_t total = 0;

  std::stringstream ss;
  auto t0 = std::chrono::steady_clock::now();
  for (auto &kv : stream_map) {
    auto stream_handle = kv.second;

    auto x = AbortStreamHandleWithLock(stream_handle, lock);
    total += x;

    std::string sep = ss.str().empty() ? "" : " | ";
    ss << sep << std::hex << "stream 0x" << kv.first << " handle 0x" << stream_handle 
       << " aborted " << std::dec << x << " commands";
  }
  auto t1 = std::chrono::steady_clock::now();

  DLOG(INFO) << "[AbortAllStreams] " << ss.str() << ", " 
            << std::chrono::duration<double, std::milli>(t1 - t0).count() << "ms";
  return total;
}

int SyncStream(uint64_t stream) {
  std::unique_lock lock{stream_mut};
  auto it = stream_map.find(stream);
  if (it == stream_map.end()) {
    LOG(FATAL) << "Stream 0x" << stream << " not registered";
  }
  return CudaXQueueSync(it->second);
}

int SyncAllStreams() {
  std::unique_lock lock{stream_mut};
  bool ret = true;
  for (auto &kv : stream_map) {
    ret &= CudaXQueueSync(kv.second);
  }
  return ret;
}

size_t GetXQueueSize(uint64_t stream) {
  std::unique_lock lock{stream_mut};
  auto stream_handle = GetRegisteredHandleWithLock(stream, lock);

  auto xqueue = CudaXQueueGet(stream_handle);
  CHECK(xqueue != nullptr) << "XQueue is NULL, stream_handle=" 
                           << stream_handle;
  return xqueue->GetSize();
}

size_t GetTotalXQueueSize() {
  std::unique_lock lock{stream_mut};
  size_t total = 0;
  for (auto &kv : stream_map) {
    auto xqueue = CudaXQueueGet(kv.second);
    CHECK(xqueue != nullptr) << "XQueue is NULL, stream_handle=" 
                             << kv.second;
    total += xqueue->GetSize();
  }
  return total;
}

int QueryRejectCudaCalls() {
  return CudaXQueueQueryReject();
}

void SetRejectCudaCalls(bool reject) {
  CudaXQueueSetReject(reject);
}

int RegisterCudaKernelLaunchPreHook(std::function<void*()> fn) {
  using CudaKernelLaunchCommand = ::xsched::hal::cuda::CudaKernelLaunchCommand;
  return CudaKernelLaunchCommand::RegisterCudaKernelLaunchPreHook(fn);
}

void StreamApply(std::function<void(uint64_t stream)> fn) {
  std::unique_lock lock{stream_mut};
  for (auto &kv : stream_map) {
    fn(kv.first);
  }
}

void GuessNcclBegin() {
  CudaGuessNcclBegin();
}

void GuessNcclEnd() {
  CudaGuessNcclEnd();
}

bool IsGuessNcclBegined() {
  return CudaIsGuessNcclBegined();
}

std::vector<uint64_t> GetNcclStreams() {
  std::vector<CUstream> nccl_streams;
  CudaNcclSteamGet(nccl_streams);
  return std::accumulate(nccl_streams.begin(), nccl_streams.end(), 
      std::vector<uint64_t>{}, 
      [](std::vector<uint64_t> &acc, CUstream stream) {
        acc.push_back(reinterpret_cast<uint64_t>(stream));
        return acc;
      });
} 


} // namesapce xsched
} // namespace sta
} // namespace colserve