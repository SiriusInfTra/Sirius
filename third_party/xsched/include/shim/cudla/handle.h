#pragma once

#include "shim/cudla/xctrl.h"
#include "preempt/xqueue/xtype.h"

namespace xsched::shim::cudla
{

cudaStream_t GetStream(preempt::XQueueHandle handle);
preempt::XQueueHandle GetXHandle(cudaStream_t stream);

} // namespace xsched::shim::cudla
