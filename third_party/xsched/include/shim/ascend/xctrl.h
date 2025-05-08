#pragma once

#include <cstdint>

#define PREEMPT_MODE_STOP_SUBMISSION    1
#define PREEMPT_MODE_DEACTIVATE         2
#define PREEMPT_MODE_INTERRUPT          3

#ifdef __cplusplus
#define ACL_CTRL_FUNC extern "C" __attribute__((visibility("default")))
#else
#define ACL_CTRL_FUNC __attribute__((visibility("default")))
#endif

typedef int32_t Prio;
typedef int32_t Bwidth;
typedef void *aclrtStream;

ACL_CTRL_FUNC uint64_t AclXQueueCreate(aclrtStream stream,
                                       int preempt_mode,
                                       int64_t queue_length,
                                       int64_t sync_interval);
ACL_CTRL_FUNC void AclXQueueDestroy(uint64_t handle);

ACL_CTRL_FUNC void AclXQueueSuspend(aclrtStream stream,
                                    bool sync_hal_queue = false);
ACL_CTRL_FUNC void AclXQueueResume(aclrtStream stream,
                                   bool drop_commands = false);

ACL_CTRL_FUNC void AclXQueueSetPriority(aclrtStream stream, Prio prio);
ACL_CTRL_FUNC void AclXQueueSetBandwidth(aclrtStream stream, Bwidth bdw);
