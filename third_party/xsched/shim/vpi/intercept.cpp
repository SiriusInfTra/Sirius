#include "hal/vpi/vpi.h"
#include "utils/symbol.h"
#include "utils/intercept.h"
#include "shim/vpi/shim.h"

using namespace xsched::shim::vpi;

DEFINE_INTERCEPT_FUNC5(XConvertImageFormat, true,
                       VPIStatus, vpiSubmitConvertImageFormat,
                       VPIStream, stream,
                       uint64_t , backend,
                       VPIImage , input,
                       VPIImage , output,
                       const VPIConvertImageFormatParams *, params);

DEFINE_INTERCEPT_FUNC9(XGaussianFilter, true,
                       VPIStatus, vpiSubmitGaussianFilter,
                       VPIStream, stream,
                       uint64_t , backend,
                       VPIImage , input,
                       VPIImage , output,
                       int32_t  , kernel_size_x,
                       int32_t  , kernel_size_y,
                       float    , sigma_x,
                       float    , sigma_y,
                       VPIBorderExtension, border);

DEFINE_INTERCEPT_FUNC7(XSubmitRescale, true,
                       VPIStatus, vpiSubmitRescale,
                       VPIStream, stream,
                       uint64_t , backend,
                       VPIImage , input,
                       VPIImage , output,
                       VPIInterpolationType , interpolation_type,
                       VPIBorderExtension, border,
                       uint64_t , flags);

DEFINE_INTERCEPT_FUNC8(XSubmitStereoDisparityEstimator, true,
                       VPIStatus, vpiSubmitStereoDisparityEstimator,
                       VPIStream , stream,
                       uint64_t  , backend,
                       VPIPayload, payload,
                       VPIImage  , left,
                       VPIImage  , right,
                       VPIImage  , disparity,
                       VPIImage  , confidence_map,
                       const VPIStereoDisparityEstimatorParams *, params);

DEFINE_INTERCEPT_FUNC2(XEventRecord, false,
                       VPIStatus, vpiEventRecord,
                       VPIEvent , event,
                       VPIStream, stream);

DEFINE_INTERCEPT_FUNC1(XEventSync, false,
                       VPIStatus, vpiEventSync,
                       VPIEvent, event);

DEFINE_INTERCEPT_FUNC1(XEventDestroy, false,
                       void, vpiEventDestroy,
                       VPIEvent, event);

DEFINE_INTERCEPT_FUNC1(XStreamSync, false,
                       VPIStatus, vpiStreamSync,
                       VPIStream, stream);

DEFINE_INTERCEPT_FUNC2(XStreamCreate, false,
                       VPIStatus, vpiStreamCreate,
                       uint64_t   , flags,
                       VPIStream *, stream);

DEFINE_INTERCEPT_FUNC1(XStreamDestroy, false,
                       void, vpiStreamDestroy,
                       VPIStream, stream);
