#pragma once

#include "shim/common/def.h"
#include "shim/common/xmanager.h"
#include "shim/cudla/handle.h"
#include "hal/cudla/cudla.h"
#include "hal/cudla/cudart.h"
#include "hal/cudla/cudla_command.h"

namespace xsched::shim::cudla
{

static inline cudaError_t SubmitCudlaCommand(
    std::shared_ptr<hal::cudla::CudlaCommand> cmd, cudaStream_t stream)
{
    preempt::XQueueHandle handle = GetXHandle(stream);
    if (XManager::Submit(cmd, handle)) return cudaSuccess;
    return cmd->EnqueueWrapper(stream);
}

DEFINE_SHIM_FUNC4(SubmitCudlaCommand, cudaStream_t,
                  hal::cudla::CudlaMemoryCommand,
                  cudaError_t   , XMemcpyAsync,
                  void *        , dst,
                  const void *  , src,
                  size_t        , count,
                  cudaMemcpyKind, kind);

cudaError_t XStreamCreate(cudaStream_t *stream);
cudaError_t XStreamCreateWithFlags(cudaStream_t *stream, unsigned int flags);
cudaError_t XStreamDestroy(cudaStream_t stream);
cudaError_t XStreamSynchronize(cudaStream_t stream);

cudaError_t XEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t XEventSynchronize(cudaEvent_t event);
cudaError_t XEventDestroy(cudaEvent_t event);

cudlaStatus XSubmitTask(cudlaDevHandle const dev_handle,
                        const cudlaTask * const tasks,
                        uint32_t const num_tasks,
                        void* const stream,
                        uint32_t const flags);

} // namespace xsched::shim::cudla
