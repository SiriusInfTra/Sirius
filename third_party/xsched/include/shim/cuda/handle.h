#pragma once

#include "shim/cuda/xctrl.h"
#include "preempt/xqueue/xtype.h"

namespace xsched::shim::cuda
{

CUstream GetStream(preempt::XQueueHandle handle);
preempt::XQueueHandle GetXHandle(CUstream stream);

} // namespace xsched::shim
