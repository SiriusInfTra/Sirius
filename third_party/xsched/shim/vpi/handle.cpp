#include "utils/common.h"
#include "shim/vpi/handle.h"

static uint64_t pid_prefix = (uint64_t)(int64_t)GetProcessId() << 48;

VPIStream xsched::shim::vpi::GetStream(xsched::preempt::XQueueHandle handle)
{
    return (VPIStream)(handle ^ pid_prefix);
}

xsched::preempt::XQueueHandle xsched::shim::vpi::GetXHandle(VPIStream stream)
{
    return (uint64_t)stream ^ pid_prefix;
}
