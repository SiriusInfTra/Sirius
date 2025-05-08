#include "utils/common.h"
#include "shim/cuda/handle.h"

static uint64_t pid_prefix = (uint64_t)(int64_t)GetProcessId() << 48;

CUstream xsched::shim::cuda::GetStream(xsched::preempt::XQueueHandle handle)
{
    return (CUstream)(handle ^ pid_prefix);
}

xsched::preempt::XQueueHandle xsched::shim::cuda::GetXHandle(CUstream stream)
{
    return (uint64_t)stream ^ pid_prefix;
}
