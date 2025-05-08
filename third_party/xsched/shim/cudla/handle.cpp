#include "utils/common.h"
#include "shim/cudla/handle.h"

static uint64_t pid_prefix = (uint64_t)(int64_t)GetProcessId() << 48;

cudaStream_t xsched::shim::cudla::GetStream(xsched::preempt::XQueueHandle handle)
{
    return (cudaStream_t)(handle ^ pid_prefix);
}

xsched::preempt::XQueueHandle xsched::shim::cudla::GetXHandle(cudaStream_t stream)
{
    return (uint64_t)stream ^ pid_prefix;
}
