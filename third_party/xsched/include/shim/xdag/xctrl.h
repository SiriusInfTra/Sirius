#pragma once

#include <cstdint>

#ifdef __cplusplus
#define XDAG_CTRL_FUNC extern "C" __attribute__((visibility("default")))
#else
#define XDAG_CTRL_FUNC __attribute__((visibility("default")))
#endif

typedef int32_t Prio;
typedef struct CUctx_st *CUcontext;
typedef struct cudlaDevHandle_t *cudlaDevHandle;

XDAG_CTRL_FUNC void XDagEnableCudaContext(CUcontext context=nullptr);
XDAG_CTRL_FUNC void XDagSetPriorityCuda(Prio priority, CUcontext context=nullptr);

XDAG_CTRL_FUNC void XDagTaskBeginCuda(CUcontext context=nullptr);
XDAG_CTRL_FUNC void XDagTaskEndCuda(CUcontext context=nullptr);

XDAG_CTRL_FUNC void XDagSuspendCuda(CUcontext context);
XDAG_CTRL_FUNC void XDagResumeCuda(CUcontext context);

XDAG_CTRL_FUNC cudlaDevHandle XDagGetLastCreatedDlaDevice();
XDAG_CTRL_FUNC void XDagEnableDlaDevice(cudlaDevHandle dev, uint64_t dev_no);
XDAG_CTRL_FUNC void XDagSetPriorityDla(Prio priority, cudlaDevHandle dev);

XDAG_CTRL_FUNC void XDagTaskBeginDla(cudlaDevHandle dev);
XDAG_CTRL_FUNC void XDagTaskEndDla(cudlaDevHandle dev);

XDAG_CTRL_FUNC void XDagSuspendDla(cudlaDevHandle dev);
XDAG_CTRL_FUNC void XDagResumeDla(cudlaDevHandle dev);

XDAG_CTRL_FUNC void XDagSuspend(uint64_t xhandle);
XDAG_CTRL_FUNC void XDagResume(uint64_t xhandle);
