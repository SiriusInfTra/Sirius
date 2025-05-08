#pragma once

#include <string>
#include <vector>

#include "utils/lib.h"
#include "utils/common.h"
#include "utils/symbol.h"
#include "hal/cuda/cuda.h"

namespace xsched::hal::cuda
{

class Driver
{
public:
    STATIC_CLASS(Driver);

    DEFINE_STATIC_SYMBOL11("cuLaunchKernel", GetSymbol,
                           CUresult    , LaunchKernel,
                           CUfunction  , function,
                           unsigned int, grid_dim_x,
                           unsigned int, grid_dim_y,
                           unsigned int, grid_dim_z,
                           unsigned int, block_dim_x,
                           unsigned int, block_dim_y,
                           unsigned int, block_dim_z,
                           unsigned int, shared_mem_bytes,
                           CUstream    , stream,
                           void **     , kernel_params,
                           void **     , extra_params);

    DEFINE_STATIC_SYMBOL3("cuLaunchHostFunc", GetSymbol,
                          CUresult, LaunchHostFunc,
                          CUstream, stream,
                          CUhostFn, host_func,
                          void *  , user_data);
    
    DEFINE_STATIC_SYMBOL4("cuMemcpyHtoDAsync_v2", GetSymbol,
                          CUresult    , MemcpyHtoDAsyncV2,
                          CUdeviceptr , dst_device,
                          const void *, src_host,
                          size_t      , byte_count,
                          CUstream    , stream);
    
    DEFINE_STATIC_SYMBOL4("cuMemcpyDtoHAsync_v2", GetSymbol,
                          CUresult   , MemcpyDtoHAsyncV2,
                          void *     , dst_host,
                          CUdeviceptr, src_device,
                          size_t     , byte_count,
                          CUstream   , stream);
    
    DEFINE_STATIC_SYMBOL4("cuMemcpyDtoDAsync_v2", GetSymbol,
                          CUresult   , MemcpyDtoDAsyncV2,
                          CUdeviceptr, dst_device,
                          CUdeviceptr, src_device,
                          size_t     , byte_count,
                          CUstream   , stream);

    DEFINE_STATIC_SYMBOL2("cuMemcpy2DAsync_v2" , GetSymbol,
                          CUresult             , Memcpy2DAsyncV2,
                          const CUDA_MEMCPY2D *, p_copy,
                          CUstream             , stream);
    
    DEFINE_STATIC_SYMBOL2("cuMemcpy3DAsync_v2" , GetSymbol,
                          CUresult             , Memcpy3DAsyncV2,
                          const CUDA_MEMCPY3D *, p_copy,
                          CUstream             , stream);
    
    DEFINE_STATIC_SYMBOL4("cuMemsetD8Async", GetSymbol,
                          CUresult     , MemsetD8Async,
                          CUdeviceptr  , dst_device,
                          unsigned char, unsigned_char,
                          size_t       , n,
                          CUstream     , stream);
    
    DEFINE_STATIC_SYMBOL4("cuMemsetD16Async", GetSymbol,
                          CUresult      , MemsetD16Async,
                          CUdeviceptr   , dst_device,
                          unsigned short, unsigned_short,
                          size_t        , n,
                          CUstream      , stream);
    
    DEFINE_STATIC_SYMBOL4("cuMemsetD32Async", GetSymbol,
                          CUresult    , MemsetD32Async,
                          CUdeviceptr , dst_device,
                          unsigned int, unsigned_int,
                          size_t      , n,
                          CUstream    , stream);
    
    DEFINE_STATIC_SYMBOL6("cuMemsetD2D8Async", GetSymbol,
                          CUresult     , MemsetD2D8Async,
                          CUdeviceptr  , dst_device,
                          size_t       , dst_pitch,
                          unsigned char, unsigned_char,
                          size_t       , width,
                          size_t       , height,
                          CUstream     , stream);
    
    DEFINE_STATIC_SYMBOL6("cuMemsetD2D16Async", GetSymbol,
                          CUresult      , MemsetD2D16Async,
                          CUdeviceptr   , dst_device,
                          size_t        , dst_pitch,
                          unsigned short, unsigned_short,
                          size_t        , width,
                          size_t        , height,
                          CUstream      , stream);
    
    DEFINE_STATIC_SYMBOL6("cuMemsetD2D32Async", GetSymbol,
                          CUresult    , MemsetD2D32Async,
                          CUdeviceptr , dst_device,
                          size_t      , dst_pitch,
                          unsigned int, unsigned_int,
                          size_t      , width,
                          size_t      , height,
                          CUstream    , stream);
    
    DEFINE_STATIC_SYMBOL2("cuMemFreeAsync", GetSymbol,
                          CUresult   , MemFreeAsync,
                          CUdeviceptr, device_ptr,
                          CUstream   , stream);

    DEFINE_STATIC_SYMBOL3("cuMemAllocAsync", GetSymbol,
                          CUresult     , MemAllocAsync,
                          CUdeviceptr *, device_ptr,
                          size_t       , byte_size,
                          CUstream     , stream);
    
    DEFINE_STATIC_SYMBOL1("cuEventQuery", GetSymbol,
                          CUresult, EventQuery,
                          CUevent , event);

    DEFINE_STATIC_SYMBOL2("cuEventRecord", GetSymbol,
                          CUresult, EventRecord,
                          CUevent , event,
                          CUstream, stream);

    DEFINE_STATIC_SYMBOL3("cuEventRecordWithFlags", GetSymbol,
                          CUresult    , EventRecordWithFlags,
                          CUevent     , event,
                          CUstream    , stream,
                          unsigned int, flags);

    DEFINE_STATIC_SYMBOL1("cuEventSynchronize", GetSymbol,
                          CUresult, EventSynchronize,
                          CUevent , event);

    DEFINE_STATIC_SYMBOL3("cuStreamWaitEvent", GetSymbol,
                          CUresult    , StreamWaitEvent,
                          CUstream    , stream,
                          CUevent     , event,
                          unsigned int, flags);

    DEFINE_STATIC_SYMBOL2("cuEventCreate", GetSymbol,
                          CUresult    , EventCreate,
                          CUevent *   , event,
                          unsigned int, flags);

    DEFINE_STATIC_SYMBOL1("cuEventDestroy", GetSymbol,
                          CUresult, EventDestroy,
                          CUevent , event);

    DEFINE_STATIC_SYMBOL1("cuEventDestroy_v2", GetSymbol,
                          CUresult, EventDestroyV2,
                          CUevent , event);

    DEFINE_STATIC_SYMBOL1("cuStreamQuery", GetSymbol,
                          CUresult, StreamQuery,
                          CUstream, stream);

    DEFINE_STATIC_SYMBOL1("cuStreamSynchronize", GetSymbol,
                          CUresult, StreamSynchronize,
                          CUstream, stream);

    DEFINE_STATIC_SYMBOL2("cuStreamCreate", GetSymbol,
                          CUresult    , StreamCreate,
                          CUstream *  , stream,
                          unsigned int, flags);

    DEFINE_STATIC_SYMBOL3("cuStreamCreateWithPriority", GetSymbol,
                          CUresult    , StreamCreateWithPriority,
                          CUstream *  , stream,
                          unsigned int, flags,
                          int         , priority);
    
    DEFINE_STATIC_SYMBOL1("cuStreamDestroy", GetSymbol,
                          CUresult, StreamDestroy,
                          CUstream, stream);
    
    DEFINE_STATIC_SYMBOL2("cuStreamGetFlags", GetSymbol,
                          CUresult      , StreamGetFlags,
                          CUstream      , stream,
                          unsigned int *, flags);
    
    DEFINE_STATIC_SYMBOL2("cuStreamGetCtx", GetSymbol,
                          CUresult   , StreamGetCtx,
                          CUstream   , stream,
                          CUcontext *, context);

    DEFINE_STATIC_SYMBOL0("cuCtxSynchronize", GetSymbol,
                          CUresult, CtxSynchronize);

    DEFINE_STATIC_SYMBOL1("cuCtxSetCurrent", GetSymbol,
                          CUresult , CtxSetCurrent,
                          CUcontext, context);
    
    DEFINE_STATIC_SYMBOL1("cuCtxGetCurrent", GetSymbol,
                          CUresult   , CtxGetCurrent,
                          CUcontext *, context);

    DEFINE_STATIC_SYMBOL1("cuCtxGetDevice", GetSymbol,
                          CUresult  , CtxGetDevice,
                          CUdevice *, device);

    DEFINE_STATIC_SYMBOL2("cuCtxGetStreamPriorityRange", GetSymbol,
                          CUresult, CtxGetStreamPriorityRange,
                          int *   , least_priority,
                          int *   , greatestPriority);
    
   DEFINE_STATIC_SYMBOL3("cuCtxCreate_v2", GetSymbol,
                         CUresult    , CtxCreateV2,
                         CUcontext * , p_ctx,
                         unsigned int, flags,
                         CUdevice    , dev);

    DEFINE_STATIC_SYMBOL2("cuGetErrorString", GetSymbol,
                          CUresult     , GetErrorString,
                          CUresult     , error,
                          const char **, str);

    DEFINE_STATIC_SYMBOL2("cuGetExportTable", GetSymbol,
                          CUresult      , GetExportTable,
                          const void ** , table,
                          const CUuuid *, table_id);

    DEFINE_STATIC_SYMBOL5("cuMemAddressReserve", GetSymbol,
                          CUresult          , MemAddressReserve,
                          CUdeviceptr *     , ptr,
                          size_t            , size,
                          size_t            , alignment,
                          CUdeviceptr       , addr,
                          unsigned long long, flags);

    DEFINE_STATIC_SYMBOL2("cuMemAddressFree", GetSymbol,
                          CUresult   , MemAddressFree,
                          CUdeviceptr, ptr,
                          size_t     , size);

    DEFINE_STATIC_SYMBOL3("cuMemGetAllocationGranularity" , GetSymbol,
                          CUresult, MemGetAllocationGranularity,
                          size_t *                        , granularity,
                          const CUmemAllocationProp *     , prop,
                          CUmemAllocationGranularity_flags, option);

    DEFINE_STATIC_SYMBOL4("cuMemSetAccess"       , GetSymbol,
                          CUresult               , MemSetAccess,
                          CUdeviceptr            , ptr,
                          size_t                 , size,
                          const CUmemAccessDesc *, desc,
                          size_t                 , count);

    DEFINE_STATIC_SYMBOL4("cuMemCreate"                 , GetSymbol,
                          CUresult                      , MemCreate,
                          CUmemGenericAllocationHandle *, handle,
                          size_t                        , size,
                          const CUmemAllocationProp *   , prop,
                          unsigned long long            , flags);

    DEFINE_STATIC_SYMBOL1("cuMemRelease"              , GetSymbol,
                          CUresult                    , MemRelease,
                          CUmemGenericAllocationHandle, handle);

    DEFINE_STATIC_SYMBOL5("cuMemMap"                  , GetSymbol,
                          CUresult                    , MemMap,
                          CUdeviceptr                 , ptr,
                          size_t                      , size,
                          size_t                      , offset,
                          CUmemGenericAllocationHandle, handle,
                          unsigned long long          , flags);

    DEFINE_STATIC_SYMBOL2("cuMemUnmap", GetSymbol,
                          CUresult   , MemUnmap,
                          CUdeviceptr, ptr,
                          size_t     , size);

    DEFINE_STATIC_SYMBOL3("cuMemAllocManaged", GetSymbol,
                          CUresult     , MemAllocManaged,
                          CUdeviceptr *, dptr,
                          size_t       , bytesize,
                          unsigned int , flags);

private:
    DEFINE_GET_SYMBOL_FUNC(GetSymbol, "libcuda.so", ENV_CUDA_DLL_PATH, {});
};

} // namespace xsched::hal::cuda
