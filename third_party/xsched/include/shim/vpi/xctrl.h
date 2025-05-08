#pragma once

#include <cstdint>

#define PREEMPT_MODE_STOP_SUBMISSION    1
#define PREEMPT_MODE_DEACTIVATE         2
#define PREEMPT_MODE_INTERRUPT          3

#ifdef __cplusplus
#define VPI_CTRL_FUNC extern "C" __attribute__((visibility("default")))
#else
#define VPI_CTRL_FUNC __attribute__((visibility("default")))
#endif

typedef int32_t Prio;
typedef int32_t Bwidth;
typedef struct VPIStreamImpl *VPIStream;

VPI_CTRL_FUNC uint64_t VpiXQueueCreate(VPIStream stream,
                                       int preempt_mode,
                                       int64_t queue_length,
                                       int64_t sync_interval);
VPI_CTRL_FUNC void VpiXQueueDestroy(uint64_t handle);

VPI_CTRL_FUNC void VpiXQueueSuspend(VPIStream stream,
                                    bool sync_hal_queue = false);
VPI_CTRL_FUNC void VpiXQueueResume(VPIStream stream,
                                   bool drop_commands = false);

VPI_CTRL_FUNC void VpiXQueueSetPriority(VPIStream stream, Prio prio);
VPI_CTRL_FUNC void VpiXQueueSetBandwidth(VPIStream stream, Bwidth bdw);
