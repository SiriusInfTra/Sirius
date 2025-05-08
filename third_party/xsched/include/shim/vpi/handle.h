#pragma once

#include "shim/vpi/xctrl.h"
#include "preempt/xqueue/xtype.h"

namespace xsched::shim::vpi
{

VPIStream GetStream(preempt::XQueueHandle handle);
preempt::XQueueHandle GetXHandle(VPIStream stream);

} // namespace xsched::shim::vpi
