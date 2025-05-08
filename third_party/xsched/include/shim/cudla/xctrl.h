#pragma once

#include <cstdint>

#define PREEMPT_MODE_STOP_SUBMISSION    1
#define PREEMPT_MODE_DEACTIVATE         2
#define PREEMPT_MODE_INTERRUPT          3

#ifdef __cplusplus
#define CUDLA_CTRL_FUNC extern "C" __attribute__((visibility("default")))
#else
#define CUDLA_CTRL_FUNC __attribute__((visibility("default")))
#endif

typedef int32_t Prio;
typedef int32_t Bwidth;
typedef struct CUstream_st *cudaStream_t;

CUDLA_CTRL_FUNC uint64_t CudlaXQueueCreate(cudaStream_t stream,
                                           int preempt_mode,
                                           int64_t queue_length,
                                           int64_t sync_interval);
CUDLA_CTRL_FUNC void CudlaXQueueDestroy(uint64_t handle);

CUDLA_CTRL_FUNC void CudlaXQueueSuspend(cudaStream_t stream,
                                        bool sync_hal_queue = false);
CUDLA_CTRL_FUNC void CudlaXQueueResume(cudaStream_t stream,
                                       bool drop_commands = false);

CUDLA_CTRL_FUNC void CudlaXQueueSetPriority(cudaStream_t stream, Prio prio);
CUDLA_CTRL_FUNC void CudlaXQueueSetBandwidth(cudaStream_t stream, Bwidth bdw);
