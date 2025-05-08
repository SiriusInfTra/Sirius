#pragma once

#include "hal/vpi/vpi.h"
#include "hal/vpi/vpi_command.h"
#include "shim/vpi/handle.h"
#include "shim/common/def.h"
#include "shim/common/xmanager.h"

namespace xsched::shim::vpi
{

static inline VPIStatus SubmitVpiCommand(
    std::shared_ptr<hal::vpi::VpiCommand> cmd, VPIStream stream)
{
    preempt::XQueueHandle handle = GetXHandle(stream);
    if (XManager::Submit(cmd, handle)) return VPI_SUCCESS;
    return cmd->EnqueueWrapper(stream);
}

DEFINE_SHIM_FUNC4(SubmitVpiCommand, VPIStream,
                  hal::vpi::VpiConvertImageFormatCommand,
                  VPIStatus, XConvertImageFormat,
                  uint64_t , backend,
                  VPIImage , input,
                  VPIImage , output,
                  const VPIConvertImageFormatParams *, params);

DEFINE_SHIM_FUNC8(SubmitVpiCommand, VPIStream,
                  hal::vpi::VpiGaussianFilterCommand,
                  VPIStatus, XGaussianFilter,
                  uint64_t , backend,
                  VPIImage , input,
                  VPIImage , output,
                  int32_t  , kernel_size_x,
                  int32_t  , kernel_size_y,
                  float    , sigma_x,
                  float    , sigma_y,
                  VPIBorderExtension, border);

DEFINE_SHIM_FUNC6(SubmitVpiCommand, VPIStream,
                  hal::vpi::VpiRescaleCommand,
                  VPIStatus, XSubmitRescale,
                  uint64_t , backend,
                  VPIImage , input,
                  VPIImage , output,
                  VPIInterpolationType , interpolation_type,
                  VPIBorderExtension, border,
                  uint64_t , flags);

DEFINE_SHIM_FUNC7(SubmitVpiCommand, VPIStream,
                  hal::vpi::VpiStereoDisparityEstimatorCommand,
                  VPIStatus, XSubmitStereoDisparityEstimator,
                  uint64_t  , backend,
                  VPIPayload, payload,
                  VPIImage  , left,
                  VPIImage  , right,
                  VPIImage  , disparity,
                  VPIImage  , confidence_map,
                  const VPIStereoDisparityEstimatorParams *, params);

VPIStatus XEventRecord(VPIEvent event, VPIStream stream);
VPIStatus XEventSync(VPIEvent event);
void XEventDestroy(VPIEvent event);

VPIStatus XStreamSync(VPIStream stream);
VPIStatus XStreamCreate(uint64_t flags, VPIStream *stream);
void XStreamDestroy(VPIStream stream);

} // namespace xsched::shim::vpi
