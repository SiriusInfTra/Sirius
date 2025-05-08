#pragma once

#include <cstdint>

namespace xsched::preempt
{

typedef uint64_t XQueueHandle;

enum XDevice
{
    kDeviceUnknown = 0,
    kDeviceVPI     = 1,
    kDeviceCUDA    = 2,
    kDeviceCUDLA   = 3,
    kDeviceAscend  = 4,
};

enum XPreemptMode
{
    kPreemptModeDisabled       = 0,
    kPreemptModeStopSubmission = 1,
    kPreemptModeDeactivate     = 2,
    kPreemptModeInterrupt      = 3,

    kPreemptModeMax,
};

enum XQueueState
{
    kQueueStateUnknown = 0,
    kQueueStateIdle    = 1,
    kQueueStateReady   = 2,
};

enum XCommandType
{
    kCommandTypeHAL                 = 0,
    kCommandTypeXQueueSynchronize   = 1,
    kCommandTypeIntervalSynchronize = 2,
    kCommandTypeXQueueDestroy       = 3,
};

enum XCommandState
{
    kCommandStateCreated      = 0,
    kCommandStateEnqueued     = 1,
    kCommandStateHalSubmmited = 2,
    kCommandStateCompleted    = 3,
};

} // namespace xsched::preempt
