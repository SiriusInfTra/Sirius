#include "PySched.h"

#include "cuda_middleware/cuda_hook.h"
#include "cuda_preempt/hal/cuda_command.h"
#include "middleware.h"
#include "cuda_middleware.h"
#include "xpreempt/hal/hal_command.h"
#include "xpreempt/utils/xassert.h"
#include "xpreempt/xqueue/xqueue.h"
#include "xpreempt/xqueue/xtype.h"
#include "xpreempt/hal/hal_queue.h"

#include <cstdint>
#include <cstdio>
#include <memory>

static std::atomic<uint64_t> global_stream_handle;
static std::atomic<uint64_t> global_stream;

uint64_t RegisterStream(uint64_t stream) {
  auto cuda_stream = reinterpret_cast<CUstream>(stream);
  global_stream = stream;
  global_stream_handle =  CudaMiddleware::EnableXSched(
    CudaMiddleware::GetHalQueueHandle(cuda_stream),
    kXPreemptModeStopSubmission, 16, 32); 
  fprintf(stderr, "[PyXsched] register stream %lu with global handle: %lu.\n", 
          stream, global_stream_handle.load()); 
  return global_stream_handle;
}

uint64_t GetRegisteredGlobalHandle() {
  return global_stream_handle;
}

uint64_t GetRegisteredGlobalStream() {
  return global_stream;
}

void UnRegisterStream() {
  CudaMiddleware::DisableXSched(global_stream_handle);
}

size_t AbortStream() {
  auto xqueue = CudaMiddleware::GetXQueue(global_stream_handle);
  XASSERT(xqueue != nullptr, "Required xqueue is not NULL, global_stream_handle=%ld.", global_stream_handle.load());
  auto remove_filter = [](std::shared_ptr<XCommand> hal_command) { 
    return std::dynamic_pointer_cast<CudaMemoryCommand>(hal_command) != nullptr || std::dynamic_pointer_cast<CudaKernelLaunchCommand>(hal_command) != nullptr;
  };
  return xqueue->Clear(remove_filter);
}

void SyncStream() {
  Middleware::Synchronize(global_stream_handle);
}

size_t GetXQueueSize() {
  auto xqueue = CudaMiddleware::GetXQueue(global_stream_handle);
  XASSERT(xqueue != nullptr, "Required xqueue is not NULL, global_stream_handle=%ld.", global_stream_handle.load());
  return xqueue->GetQueueSize();
}

void SetBlockCudaCalls(uint64_t handle, bool enable) {
  auto xqueue = CudaMiddleware::GetXQueue(global_stream_handle);
  XASSERT(xqueue != nullptr, "Required xqueue is not NULL, global_stream_handle=%ld.", global_stream_handle.load());
  xqueue->SetBlockCudaCalls(enable);
}

bool IsBlockCudaCalls_v2() {
  return IsBlockCudaCallsInternal();
}

void SetBlockCudaCalls_v2(bool block) {
  SetBlockCudaCallsInternal(block);
}

void StopSubmit(bool synchronous) {
  CudaMiddleware::Preempt(global_stream_handle, synchronous);
}

void ResumeSubmit() {
  CudaMiddleware::Resume(global_stream_handle);
}

bool RegisterCudaKernelLaunchPreHook(std::function<void*()> fn) {
  return XCommand::RegisterCudaKernelLaunchPreHook(fn);
}