#pragma once

#include <cstdint>

#include "shim/common/xmanager.h"

#define PREEMPT_MODE_STOP_SUBMISSION    1
#define PREEMPT_MODE_DEACTIVATE         2
#define PREEMPT_MODE_INTERRUPT          3

#ifdef __cplusplus
#define CUDA_CTRL_FUNC extern "C" __attribute__((visibility("default")))
#else
#define CUDA_CTRL_FUNC __attribute__((visibility("default")))
#endif

typedef int32_t Prio;
typedef int32_t Bwidth;
typedef struct CUstream_st* CUstream;

CUDA_CTRL_FUNC uint64_t CudaXQueueCreate(CUstream stream,
                                         int preempt_mode,
                                         int64_t queue_length,
                                         int64_t sync_interval);
CUDA_CTRL_FUNC void CudaXQueueDestroy(uint64_t handle);

CUDA_CTRL_FUNC xsched::preempt::XQueue* CudaXQueueGet(uint64_t handle);

CUDA_CTRL_FUNC void CudaXQueuePreempt(CUstream stream,
                                      bool sync_hal_queue = false);
CUDA_CTRL_FUNC void CudaXQueueResume(CUstream stream,
                                     bool drop_commands = false);

CUDA_CTRL_FUNC void CudaSyncBlockingXQueues();

CUDA_CTRL_FUNC void CudaXQueueSetPriority(CUstream stream, Prio priority);
CUDA_CTRL_FUNC void CudaXQueueSetBandwidth(CUstream stream, Bwidth bdw);

CUDA_CTRL_FUNC void CudaXQueueSetReject(bool reject);
CUDA_CTRL_FUNC bool CudaXQueueQueryReject();

CUDA_CTRL_FUNC bool CudaXQueueSync(uint64_t handle);

//////////////////////////////////////////////////////////
// extra functions to get apllication related information

CUDA_CTRL_FUNC void CudaGuessNcclBegin();
CUDA_CTRL_FUNC void CudaGuessNcclEnd();
CUDA_CTRL_FUNC bool CudaIsGuessNcclBegined();
CUDA_CTRL_FUNC void CudaNcclSteamGet(std::vector<CUstream> &nccl_stream_ret);

