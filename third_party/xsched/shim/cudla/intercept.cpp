#include "utils/symbol.h"
#include "utils/intercept.h"
#include "shim/cudla/shim.h"
#include "hal/cudla/cudla.h"
#include "hal/cudla/cudart.h"
#include "hal/cudla/driver.h"

using namespace xsched::hal::cudla;
using namespace xsched::shim::cudla;

DEFINE_INTERCEPT_FUNC1(XStreamCreate, false,
                       cudaError_t, cudaStreamCreate,
                       cudaStream_t *, p_stream);

DEFINE_INTERCEPT_FUNC2(XStreamCreateWithFlags, false,
                       cudaError_t, cudaStreamCreateWithFlags,
                       cudaStream_t *, p_stream,
                       unsigned int  , flags);

DEFINE_INTERCEPT_FUNC1(XStreamDestroy, false,
                       cudaError_t , cudaStreamDestroy,
                       cudaStream_t, stream);

DEFINE_INTERCEPT_FUNC1(XStreamSynchronize, false,
                       cudaError_t, cudaStreamSynchronize,
                       cudaStream_t, stream);

DEFINE_INTERCEPT_FUNC2(XEventRecord, false,
                       cudaError_t, cudaEventRecord,
                       cudaEvent_t , event,
                       cudaStream_t, stream);

DEFINE_INTERCEPT_FUNC1(XEventSynchronize, false,
                       cudaError_t, cudaEventSynchronize,
                       cudaEvent_t, event);

DEFINE_INTERCEPT_FUNC1(XEventDestroy, false,
                       cudaError_t, cudaEventDestroy,
                       cudaEvent_t, event);

DEFINE_INTERCEPT_FUNC5(XMemcpyAsync, false,
                       cudaError_t, cudaMemcpyAsync,
                       void *        , dst,
                       const void *  , src,
                       size_t        , count,
                       cudaMemcpyKind, kind,
                       cudaStream_t  , stream);

DEFINE_INTERCEPT_FUNC5(XSubmitTask, false,
                       cudlaStatus, cudlaSubmitTask,
                       cudlaDevHandle const   , dev_handle,
                       const cudlaTask * const, ptr_to_tasks,
                       uint32_t const         , num_tasks,
                       void * const           , stream,
                       uint32_t const         , flags);
