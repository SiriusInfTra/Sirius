#pragma once

#include <string>

#include "utils/common.h"
#include "utils/symbol.h"
#include "hal/cudla/cudla.h"
#include "hal/cudla/cudart.h"

namespace xsched::hal::cudla
{

class DlaDriver
{
public:
    STATIC_CLASS(DlaDriver);

    DEFINE_STATIC_SYMBOL1("cudlaGetVersion", GetSymbol,
                          cudlaStatus     , GetVersion,
                          uint64_t * const, version);
    
    DEFINE_STATIC_SYMBOL1("cudlaDeviceGetCount", GetSymbol,
                          cudlaStatus     , DeviceGetCount,
                          uint64_t * const, p_num_devices);
    
    DEFINE_STATIC_SYMBOL3("cudlaCreateDevice", GetSymbol,
                          cudlaStatus           , CreateDevice,
                          uint64_t const        , device,
                          cudlaDevHandle * const, dev_handle,
                          uint32_t const        , flags);
    
    DEFINE_STATIC_SYMBOL5("cudlaMemRegister", GetSymbol,
                          cudlaStatus           , MemRegister,
                          cudlaDevHandle const  , dev_handle,
                          const uint64_t * const, ptr,
                          size_t const          , size,
                          uint64_t ** const     , dev_ptr,
                          uint32_t const        , flags);
    
    DEFINE_STATIC_SYMBOL5("cudlaModuleLoadFromMemory", GetSymbol,
                          cudlaStatus          , ModuleLoadFromMemory,
                          cudlaDevHandle const , dev_handle,
                          const uint8_t * const, p_module,
                          size_t const         , module_size,
                          cudlaModule * const  , h_module,
                          uint32_t const       , flags);
    
    DEFINE_STATIC_SYMBOL3("cudlaModuleGetAttributes", GetSymbol,
                          cudlaStatus                   , ModuleGetAttributes,
                          cudlaModule const             , h_module,
                          cudlaModuleAttributeType const, attr_type,
                          cudlaModuleAttribute * const  , attribute);
    
    DEFINE_STATIC_SYMBOL2("cudlaModuleUnload", GetSymbol,
                          cudlaStatus      , ModuleUnload,
                          cudlaModule const, h_module,
                          uint32_t const   , flags);
    
    DEFINE_STATIC_SYMBOL5("cudlaSubmitTask", GetSymbol,
                          cudlaStatus            , SubmitTask,
                          cudlaDevHandle const   , dev_handle,
                          const cudlaTask * const, ptr_to_tasks,
                          uint32_t const         , num_tasks,
                          void * const           , stream,
                          uint32_t const         , flags);
    
    DEFINE_STATIC_SYMBOL3("cudlaDeviceGetAttribute", GetSymbol,
                          cudlaStatus                , DeviceGetAttribute,
                          cudlaDevHandle const       , dev_handle,
                          cudlaDevAttributeType const, attrib,
                          cudlaDevAttribute * const  , p_attribute);
    
    DEFINE_STATIC_SYMBOL2("cudlaMemUnregister", GetSymbol,
                          cudlaStatus           , MemUnregister,
                          cudlaDevHandle const  , dev_handle,
                          const uint64_t * const, dev_ptr);
    
    DEFINE_STATIC_SYMBOL1("cudlaGetLastError", GetSymbol,
                          cudlaStatus         , GetLastError,
                          cudlaDevHandle const, dev_handle);
    
    DEFINE_STATIC_SYMBOL1("cudlaDestroyDevice", GetSymbol,
                          cudlaStatus         , DestroyDevice,
                          cudlaDevHandle const, dev_handle);
    
    DEFINE_STATIC_SYMBOL4("cudlaImportExternalMemory", GetSymbol,
                          cudlaStatus         , ImportExternalMemory,
                          cudlaDevHandle const, dev_handle,
                          const cudlaExternalMemoryHandleDesc * const, desc,
                          uint64_t ** const   , dev_ptr,
                          uint32_t const      , flags);
    
    DEFINE_STATIC_SYMBOL2("cudlaGetNvSciSyncAttributes", GetSymbol,
                          cudlaStatus     , GetNvSciSyncAttributes,
                          uint64_t * const, attr_list,
                          uint32_t const  , flags);
    
    DEFINE_STATIC_SYMBOL4("cudlaImportExternalSemaphore", GetSymbol,
                          cudlaStatus         , ImportExternalSemaphore,
                          cudlaDevHandle const, dev_handle,
                          const cudlaExternalSemaphoreHandleDesc * const, desc,
                          uint64_t ** const   , dev_ptr,
                          uint32_t const      , flags);
    
    DEFINE_STATIC_SYMBOL2("cudlaSetTaskTimeoutInMs", GetSymbol,
                          cudlaStatus         , SetTaskTimeoutInMs,
                          cudlaDevHandle const, dev_handle,
                          uint32_t const      , timeout);

private:
    DEFINE_GET_SYMBOL_FUNC(GetSymbol, "libcudla.so", ENV_CUDLA_DLL_PATH,
                           {"/usr/local/cuda/lib64"});
};

class RtDriver
{
public:
    STATIC_CLASS(RtDriver);

    DEFINE_STATIC_SYMBOL5("cudaMemcpyAsync", GetSymbol,
                          cudaError_t   , MemcpyAsync,
                          void *        , dst,
                          const void *  , src,
                          size_t        , count,
                          cudaMemcpyKind, kind,
                          cudaStream_t  , stream);
    
    DEFINE_STATIC_SYMBOL1("cudaStreamCreate", GetSymbol,
                          cudaError_t   , StreamCreate,
                          cudaStream_t *, stream);
    
    DEFINE_STATIC_SYMBOL2("cudaStreamCreateWithFlags", GetSymbol,
                          cudaError_t   , StreamCreateWithFlags,
                          cudaStream_t *, stream,
                          unsigned int  , flags);
    
    DEFINE_STATIC_SYMBOL1("cudaStreamDestroy", GetSymbol,
                          cudaError_t , StreamDestroy,
                          cudaStream_t, stream);
    
    DEFINE_STATIC_SYMBOL1("cudaStreamSynchronize", GetSymbol,
                          cudaError_t , StreamSynchronize,
                          cudaStream_t, stream);
    
    DEFINE_STATIC_SYMBOL2("cudaEventCreateWithFlags", GetSymbol,
                          cudaError_t  , EventCreateWithFlags,
                          cudaEvent_t *, event,
                          unsigned int , flags);
    
    DEFINE_STATIC_SYMBOL1("cudaEventDestroy", GetSymbol,
                          cudaError_t, EventDestroy,
                          cudaEvent_t, event);
    
    DEFINE_STATIC_SYMBOL2("cudaEventRecord", GetSymbol,
                          cudaError_t , EventRecord,
                          cudaEvent_t , event,
                          cudaStream_t, stream);
    
    DEFINE_STATIC_SYMBOL1("cudaEventSynchronize", GetSymbol,
                          cudaError_t, EventSynchronize,
                          cudaEvent_t, event);

private:
    DEFINE_GET_SYMBOL_FUNC(GetSymbol, "libcudart.so", ENV_CUDART_DLL_PATH,
                           {"/usr/local/cuda/lib64"});
};

} // namespace xsched::hal::cudla
