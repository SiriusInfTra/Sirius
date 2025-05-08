#pragma once

#include "shim/common/def.h"
#include "shim/common/xmanager.h"
#include "shim/cuda/handle.h"
#include "hal/cuda/cuda.h"
#include "hal/cuda/driver.h"
#include "hal/cuda/cuda_queue.h"
#include "hal/cuda/cuda_command.h"

namespace xsched::shim::cuda
{

static inline CUresult SubmitCudaCommand(
    std::shared_ptr<hal::cuda::CudaCommand> cmd, CUstream stream)
{
    preempt::XQueueHandle handle = GetXHandle(stream);
    if (XManager::Submit(cmd, handle)) return CUDA_SUCCESS;
    return cmd->EnqueueWrapper(stream);
}

////////////////////////////// kernel related //////////////////////////////
CUresult XLaunchKernel(CUfunction f,
                       unsigned int grid_dim_x,
                       unsigned int grid_dim_y,
                       unsigned int grid_dim_z,
                       unsigned int block_dim_x,
                       unsigned int block_dim_y,
                       unsigned int block_dim_z,
                       unsigned int shared_mem_bytes,
                       CUstream stream,
                       void **kernel_params,
                       void **extra);

DEFINE_SHIM_FUNC2(SubmitCudaCommand, CUstream,
                  hal::cuda::CudaHostFuncCommand,
                  CUresult    , XLaunchHostFunc,
                  CUhostFn    , host_func,
                  void *      , user_data);

////////////////////////////// memory related //////////////////////////////
DEFINE_SHIM_FUNC3(SubmitCudaCommand, CUstream,
                  hal::cuda::CudaMemcpyHtoDV2Command,
                  CUresult    , XMemcpyHtoDAsync_v2,
                  CUdeviceptr , dst_dev,
                  const void *, src_host,
                  size_t      , byte_cnt);

DEFINE_SHIM_FUNC3(SubmitCudaCommand, CUstream,
                  hal::cuda::CudaMemcpyDtoHV2Command,
                  CUresult   , XMemcpyDtoHAsync_v2,
                  void *     , dst_host,
                  CUdeviceptr, src_dev,
                  size_t     , byte_cnt);

DEFINE_SHIM_FUNC3(SubmitCudaCommand, CUstream,
                  hal::cuda::CudaMemcpyDtoDV2Command,
                  CUresult   , XMemcpyDtoDAsync_v2,
                  CUdeviceptr, dst_dev,
                  CUdeviceptr, src_dev,
                  size_t     , byte_cnt);

DEFINE_SHIM_FUNC1(SubmitCudaCommand, CUstream,
                  hal::cuda::CudaMemcpy2DV2Command,
                  CUresult            , XMemcpy2DAsync_v2,
                  const CUDA_MEMCPY2D*, p_copy);

DEFINE_SHIM_FUNC1(SubmitCudaCommand, CUstream,
                  hal::cuda::CudaMemcpy3DV2Command,
                  CUresult             , XMemcpy3DAsync_v2,
                  const CUDA_MEMCPY3D *, p_copy);

DEFINE_SHIM_FUNC3(SubmitCudaCommand, CUstream,
                  hal::cuda::CudaMemsetD8Command,
                  CUresult     , XMemsetD8Async,
                  CUdeviceptr  , dst_dev,
                  unsigned char, uc,
                  size_t       , n);

DEFINE_SHIM_FUNC3(SubmitCudaCommand, CUstream,
                  hal::cuda::CudaMemsetD16Command,
                  CUresult      , XMemsetD16Async,
                  CUdeviceptr   , dst_dev,
                  unsigned short, us,
                  size_t        , n);

DEFINE_SHIM_FUNC3(SubmitCudaCommand, CUstream,
                  hal::cuda::CudaMemsetD32Command,
                  CUresult    , XMemsetD32Async,
                  CUdeviceptr , dst_dev,
                  unsigned int, ui,
                  size_t      , n);

DEFINE_SHIM_FUNC5(SubmitCudaCommand, CUstream,
                  hal::cuda::CudaMemsetD2D8Command,
                  CUresult     , XMemsetD2D8Async,
                  CUdeviceptr  , dst_dev,
                  size_t       , dst_pitch,
                  unsigned char, uc,
                  size_t       , width,
                  size_t       , height);

DEFINE_SHIM_FUNC5(SubmitCudaCommand, CUstream,
                  hal::cuda::CudaMemsetD2D16Command,
                  CUresult      , XMemsetD2D16Async,
                  CUdeviceptr   , dst_dev,
                  size_t        , dst_pitch,
                  unsigned short, us,
                  size_t        , width,
                  size_t        , height);

DEFINE_SHIM_FUNC5(SubmitCudaCommand, CUstream,
                  hal::cuda::CudaMemsetD2D32Command,
                  CUresult    , XMemsetD2D32Async,
                  CUdeviceptr , dst_dev,
                  size_t      , dst_pitch,
                  unsigned int, ui,
                  size_t      , width,
                  size_t      , height);

DEFINE_SHIM_FUNC1(SubmitCudaCommand, CUstream,
                  hal::cuda::CudaMemoryFreeCommand,
                  CUresult   , XMemFreeAsync,
                  CUdeviceptr, dptr);

DEFINE_SHIM_FUNC2(SubmitCudaCommand, CUstream,
                  hal::cuda::CudaMemoryAllocCommand,
                  CUresult     , XMemAllocAsync,
                  CUdeviceptr *, dptr,
                  size_t       , bytesize);

////////////////////////////// event related //////////////////////////////
CUresult XEventQuery(CUevent event);
CUresult XEventRecord(CUevent event, CUstream stream);
CUresult XEventRecordWithFlags(CUevent event, CUstream stream,
                               unsigned int flags);
CUresult XEventSynchronize(CUevent event);
CUresult XStreamWaitEvent(CUstream stream, CUevent event, unsigned int flags);
CUresult XEventDestroy(CUevent event);
CUresult XEventDestroy_v2(CUevent event);

////////////////////////////// stream related //////////////////////////////
CUresult XStreamSynchronize(CUstream stream);
CUresult XStreamQuery(CUstream stream);
CUresult XCtxSynchronize();

CUresult XStreamCreate(CUstream *stream, unsigned int flags);
CUresult XStreamCreateWithPriority(CUstream *stream, unsigned int flags,
                                   int priority);

} // namespace xsched::shim::cuda
