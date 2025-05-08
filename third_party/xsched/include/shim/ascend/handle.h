#pragma once

#include "shim/ascend/xctrl.h"
#include "preempt/xqueue/xtype.h"

namespace xsched::shim::ascend
{

aclrtStream GetStream(preempt::XQueueHandle handle);
preempt::XQueueHandle GetXHandle(aclrtStream stream);

} // namespace xsched::ascend
