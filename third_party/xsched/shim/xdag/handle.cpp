#include "utils/common.h"
#include "shim/xdag/handle.h"

static uint64_t pid_prefix = (uint64_t)(int64_t)GetProcessId() << 48;

CUcontext xsched::shim::xdag::GetCudaContext(preempt::XQueueHandle handle)
{
    return (CUcontext)(handle ^ pid_prefix);
}

xsched::preempt::XQueueHandle
xsched::shim::xdag::GetXHandleForCuda(CUcontext context)
{
    return (uint64_t)context ^ pid_prefix;
}

cudlaDevHandle xsched::shim::xdag::GetDlaDevice(preempt::XQueueHandle handle)
{
    return (cudlaDevHandle)(handle ^ pid_prefix);
}

xsched::preempt::XQueueHandle 
xsched::shim::xdag::GetXHandleForDla(cudlaDevHandle device)
{
    return (uint64_t)device ^ pid_prefix;
}
