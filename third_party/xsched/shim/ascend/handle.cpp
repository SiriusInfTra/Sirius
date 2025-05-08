#include "utils/common.h"
#include "shim/ascend/handle.h"

static uint64_t pid_prefix = (uint64_t)(int64_t)GetProcessId() << 48;

aclrtStream xsched::shim::ascend::GetStream(xsched::preempt::XQueueHandle handle)
{
    return (aclrtStream)(handle ^ pid_prefix);
}

xsched::preempt::XQueueHandle xsched::shim::ascend::GetXHandle(aclrtStream stream)
{
    return (uint64_t)stream ^ pid_prefix;
}
