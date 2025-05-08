#pragma once

#include "hal/cuda/cuda.h"
#include "hal/cudla/cudla.h"
#include "shim/common/def.h"

namespace xsched::shim::xdag
{

CUresult XCtxCreateV2(CUcontext *pctx, unsigned int flags, CUdevice dev);
cudlaStatus XCreateDevice(uint64_t device, cudlaDevHandle *dev_handle, uint32_t flags);

} // namespace xsched::shim::xdag
