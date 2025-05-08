#pragma once

#include <string>

#include "hal/vpi/vpi.h"
#include "utils/common.h"
#include "utils/symbol.h"

namespace xsched::hal::vpi
{

class Driver
{
public:
    STATIC_CLASS(Driver);

    DEFINE_STATIC_SYMBOL2("vpiGetLastStatusMessage", GetSymbol,
                          VPIStatus, GetLastStatusMessage,
                          char * , msg_buffer,
                          int32_t, len_buffer);
    
    DEFINE_STATIC_SYMBOL1("vpiStatusGetName", GetSymbol,
                          const char *, StatusGetName,
                          VPIStatus, status);

    DEFINE_STATIC_SYMBOL2("vpiStreamCreate", GetSymbol,
                          VPIStatus, StreamCreate,
                          uint64_t   , flags,
                          VPIStream *, stream);

    DEFINE_STATIC_SYMBOL1("vpiStreamDestroy", GetSymbol,
                          void, StreamDestroy,
                          VPIStream, stream);

    DEFINE_STATIC_SYMBOL1("vpiStreamSync", GetSymbol,
                          VPIStatus, StreamSync,
                          VPIStream, stream);

    DEFINE_STATIC_SYMBOL2("vpiEventCreate", GetSymbol,
                          VPIStatus, EventCreate,
                          uint64_t  , flags,
                          VPIEvent *, event);

    DEFINE_STATIC_SYMBOL1("vpiEventDestroy", GetSymbol,
                          void    , EventDestroy,
                          VPIEvent, event);

    DEFINE_STATIC_SYMBOL2("vpiEventRecord", GetSymbol,
                          VPIStatus, EventRecord,
                          VPIEvent , event,
                          VPIStream, stream);

    DEFINE_STATIC_SYMBOL1("vpiEventSync", GetSymbol,
                          VPIStatus, EventSync,
                          VPIEvent, event);

    DEFINE_STATIC_SYMBOL5("vpiSubmitConvertImageFormat", GetSymbol,
                          VPIStatus, SubmitConvertImageFormat,
                          VPIStream, stream,
                          uint64_t , backend,
                          VPIImage , input,
                          VPIImage , output,
                          const VPIConvertImageFormatParams *, params);

    DEFINE_STATIC_SYMBOL9("vpiSubmitGaussianFilter", GetSymbol,
                          VPIStatus, SubmitGaussianFilter,
                          VPIStream, stream,
                          uint64_t , backend,
                          VPIImage , input,
                          VPIImage , output,
                          int32_t  , kernel_size_x,
                          int32_t  , kernel_size_y,
                          float    , sigma_x,
                          float    , sigma_y,
                          VPIBorderExtension, border);

    DEFINE_STATIC_SYMBOL7("vpiSubmitRescale", GetSymbol,
                          VPIStatus, SubmitRescale,
                          VPIStream, stream,
                          uint64_t , backend,
                          VPIImage , input,
                          VPIImage , output,
                          VPIInterpolationType, interpolation_type,
                          VPIBorderExtension  , border,
                          uint64_t , flags);

    DEFINE_STATIC_SYMBOL8("vpiSubmitStereoDisparityEstimator", GetSymbol,
                          VPIStatus, SubmitStereoDisparityEstimator,
                          VPIStream , stream,
                          uint64_t  , backend,
                          VPIPayload, payload,
                          VPIImage  , left,
                          VPIImage  , right,
                          VPIImage  , disparity,
                          VPIImage  , confidence_map,
                          const VPIStereoDisparityEstimatorParams *, params);

private:
    DEFINE_GET_SYMBOL_FUNC(GetSymbol, "libnvvpi.so", ENV_VPI_DLL_PATH,
                           {"/opt/nvidia/vpi2/lib64"});
};

} // namespace xsched::hal::vpi
