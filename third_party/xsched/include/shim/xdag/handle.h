#pragma once

#include "shim/xdag/xctrl.h"
#include "preempt/xqueue/xtype.h"

namespace xsched::shim::xdag
{

CUcontext GetCudaContext(preempt::XQueueHandle handle);
preempt::XQueueHandle GetXHandleForCuda(CUcontext context);

cudlaDevHandle GetDlaDevice(preempt::XQueueHandle handle);
preempt::XQueueHandle GetXHandleForDla(cudlaDevHandle device);

} // namespace xsched::shim::xdag
