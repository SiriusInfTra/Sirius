// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 17:19:12 on Sun, May 29, 2022
//
// Description: auto generate 378 apis, manually add 33 apis and delete 1 api

#include <dlfcn.h>

#include "utils/log.h"
#include "utils/lib.h"
#include "utils/common.h"
#include "utils/xassert.h"
#include "hal/cuda/cuda.h"
#include "shim/cuda/shim.h"

#include "shim/cuda/extra.h"

#define ENABLE_INTERCEPT_DEBUG_LOG 0

#define CHECK_CUDA_XQUEUE_REJECT_CALLS if (CudaXQueueQueryReject()) { \
    return CUDA_SUCCESS; \
 } \

#if ENABLE_INTERCEPT_DEBUG_LOG
#define INTERCEPT_LOG(...) XDEBG(...)
#else
#define INTERCEPT_LOG(...)
#endif


static inline void *GetCudaSymbol(const char *name) {
    static const std::string dll_path = xsched::utils::FindLibrary("libcuda.so", ENV_CUDA_DLL_PATH, {});
    static void *dll_handle = dlopen(dll_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    XASSERT(dll_handle != nullptr, "fail to dlopen %s", dll_path.c_str());
    void *symbol = dlsym(dll_handle, name);
    XASSERT(symbol != nullptr, "fail to get symbol %s", name);
    return symbol;
}

/////////////// added by Weihang Shen to support TensorRT8.6 on CUDA12 ///////////////
EXPORT_C_FUNC CUresult cuModuleGetLoadingMode(CUmoduleLoadingMode *mode) {
    INTERCEPT_LOG("intercepted cuModuleGetLoadingMode");
    using func_ptr = CUresult (*)(CUmoduleLoadingMode *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuModuleGetLoadingMode"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(mode);
}

EXPORT_C_FUNC CUresult 
cuLibraryLoadData(CUlibrary *library, const void *code,
                  CUjit_option *jitOptions, void **jitOptionsValues, unsigned int numJitOptions,
                  CUlibraryOption *libraryOptions, void **libraryOptionValues, unsigned int numLibraryOptions) {
    INTERCEPT_LOG("intercepted cuLibraryLoadData");
    using func_ptr = CUresult (*)(CUlibrary *, const void *, CUjit_option *, void **, unsigned int, CUlibraryOption *, void **, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuLibraryLoadData"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(library, code, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions);
}

EXPORT_C_FUNC CUresult 
cuLibraryLoadFromFile(CUlibrary *library, const char *fileName,
                      CUjit_option *jitOptions, void **jitOptionsValues, unsigned int numJitOptions,
                      CUlibraryOption *libraryOptions, void **libraryOptionValues, unsigned int numLibraryOptions) {
    INTERCEPT_LOG("intercepted cuLibraryLoadFromFile");
    using func_ptr = CUresult (*)(CUlibrary *, const char *, CUjit_option *, void **, unsigned int, CUlibraryOption *, void **, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuLibraryLoadFromFile"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(library, fileName, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions);
}

EXPORT_C_FUNC CUresult cuLibraryUnload(CUlibrary library) {
    INTERCEPT_LOG("intercepted cuLibraryUnload");
    using func_ptr = CUresult (*)(CUlibrary);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuLibraryUnload"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(library);
}

EXPORT_C_FUNC CUresult cuLibraryGetKernel(CUkernel *pKernel, CUlibrary library, const char *name) {
    INTERCEPT_LOG("intercepted cuLibraryGetKernel");
    using func_ptr = CUresult (*)(CUkernel *, CUlibrary, const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuLibraryGetKernel"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pKernel, library, name);
}

EXPORT_C_FUNC CUresult cuLibraryGetModule(CUmodule *pMod, CUlibrary library) {
    INTERCEPT_LOG("intercepted cuLibraryGetModule");
    using func_ptr = CUresult (*)(CUmodule *, CUlibrary);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuLibraryGetModule"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pMod, library);
}

EXPORT_C_FUNC CUresult cuLibraryGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUlibrary library, const char *name) {
    INTERCEPT_LOG("intercepted cuLibraryGetGlobal");
    using func_ptr = CUresult (*)(CUdeviceptr *, size_t *, CUlibrary, const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuLibraryGetGlobal"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dptr, bytes, library, name);
}

EXPORT_C_FUNC CUresult cuLibraryGetManaged(CUdeviceptr *dptr, size_t *bytes, CUlibrary library, const char *name) {
    INTERCEPT_LOG("intercepted cuLibraryGetManaged");
    using func_ptr = CUresult (*)(CUdeviceptr *, size_t *, CUlibrary, const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuLibraryGetManaged"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dptr, bytes, library, name);
}

EXPORT_C_FUNC CUresult cuLibraryGetUnifiedFunction(void **fptr, CUlibrary library, const char *symbol) {
    INTERCEPT_LOG("intercepted cuLibraryGetUnifiedFunction");
    using func_ptr = CUresult (*)(void **, CUlibrary, const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuLibraryGetUnifiedFunction"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(fptr, library, symbol);
}
/////////////// added by Weihang Shen to support TensorRT8.6 on CUDA12 end ///////////////

EXPORT_C_FUNC CUresult cuGetErrorString(CUresult error, const char **pStr) {
    INTERCEPT_LOG("intercepted cuGetErrorString");
    using func_ptr = CUresult (*)(CUresult, const char **);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGetErrorString"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(error, pStr);
}

EXPORT_C_FUNC CUresult cuGetErrorName(CUresult error, const char **pStr) {
    INTERCEPT_LOG("intercepted cuGetErrorName");
    using func_ptr = CUresult (*)(CUresult, const char **);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGetErrorName"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(error, pStr);
}

EXPORT_C_FUNC CUresult cuInit(unsigned int Flags) {
    INTERCEPT_LOG("intercepted cuInit");
    using func_ptr = CUresult (*)(unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuInit"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(Flags);
}

EXPORT_C_FUNC CUresult cuDriverGetVersion(int *driverVersion) {
    INTERCEPT_LOG("intercepted cuDriverGetVersion");
    using func_ptr = CUresult (*)(int *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuDriverGetVersion"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(driverVersion);
}

EXPORT_C_FUNC CUresult cuDeviceGet(CUdevice *device, int ordinal) {
    INTERCEPT_LOG("intercepted cuDeviceGet");
    using func_ptr = CUresult (*)(CUdevice *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuDeviceGet"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(device, ordinal);
}

EXPORT_C_FUNC CUresult cuDeviceGetCount(int *count) {
    INTERCEPT_LOG("intercepted cuDeviceGetCount");
    using func_ptr = CUresult (*)(int *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuDeviceGetCount"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(count);
}

EXPORT_C_FUNC CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
    INTERCEPT_LOG("intercepted cuDeviceGetName");
    using func_ptr = CUresult (*)(char *, int, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuDeviceGetName"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(name, len, dev);
}

EXPORT_C_FUNC CUresult cuDeviceGetUuid(CUuuid *uuid, CUdevice dev) {
    INTERCEPT_LOG("intercepted cuDeviceGetUuid");
    using func_ptr = CUresult (*)(CUuuid *, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuDeviceGetUuid"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(uuid, dev);
}

EXPORT_C_FUNC CUresult cuDeviceGetUuid_v2(CUuuid *uuid, CUdevice dev) {
    INTERCEPT_LOG("intercepted cuDeviceGetUuid_v2");
    using func_ptr = CUresult (*)(CUuuid *, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuDeviceGetUuid_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(uuid, dev);
}

EXPORT_C_FUNC CUresult cuDeviceGetLuid(char *luid, unsigned int *deviceNodeMask, CUdevice dev) {
    INTERCEPT_LOG("intercepted cuDeviceGetLuid");
    using func_ptr = CUresult (*)(char *, unsigned int *, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuDeviceGetLuid"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(luid, deviceNodeMask, dev);
}

EXPORT_C_FUNC CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
    INTERCEPT_LOG("intercepted cuDeviceTotalMem");
    using func_ptr = CUresult (*)(size_t *, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuDeviceTotalMem"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(bytes, dev);
}

// manually add
EXPORT_C_FUNC CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev) {
    INTERCEPT_LOG("intercepted cuDeviceTotalMem_v2");
    using func_ptr = CUresult (*)(size_t *, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuDeviceTotalMem_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(bytes, dev);
}

EXPORT_C_FUNC CUresult cuDeviceGetTexture1DLinearMaxWidth(size_t *maxWidthInElements,
                                                          CUarray_format format, unsigned numChannels,
                                                          CUdevice dev) {
    INTERCEPT_LOG("intercepted cuDeviceGetTexture1DLinearMaxWidth");
    using func_ptr = CUresult (*)(size_t *, CUarray_format, unsigned, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuDeviceGetTexture1DLinearMaxWidth"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(maxWidthInElements, format, numChannels, dev);
}

EXPORT_C_FUNC CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) {
    INTERCEPT_LOG("intercepted cuDeviceGetAttribute");
    using func_ptr = CUresult (*)(int *, CUdevice_attribute, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuDeviceGetAttribute"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pi, attrib, dev);
}

EXPORT_C_FUNC CUresult cuDeviceGetNvSciSyncAttributes(void *nvSciSyncAttrList, CUdevice dev, int flags) {
    INTERCEPT_LOG("intercepted cuDeviceGetNvSciSyncAttributes");
    using func_ptr = CUresult (*)(void *, CUdevice, int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuDeviceGetNvSciSyncAttributes"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(nvSciSyncAttrList, dev, flags);
}

EXPORT_C_FUNC CUresult cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool) {
    INTERCEPT_LOG("intercepted cuDeviceSetMemPool");
    using func_ptr = CUresult (*)(CUdevice, CUmemoryPool);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuDeviceSetMemPool"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dev, pool);
}

EXPORT_C_FUNC CUresult cuDeviceGetMemPool(CUmemoryPool *pool, CUdevice dev) {
    INTERCEPT_LOG("intercepted cuDeviceGetMemPool");
    using func_ptr = CUresult (*)(CUmemoryPool *, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuDeviceGetMemPool"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pool, dev);
}

EXPORT_C_FUNC CUresult cuDeviceGetDefaultMemPool(CUmemoryPool *pool_out, CUdevice dev) {
    INTERCEPT_LOG("intercepted cuDeviceGetDefaultMemPool");
    using func_ptr = CUresult (*)(CUmemoryPool *, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuDeviceGetDefaultMemPool"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pool_out, dev);
}

EXPORT_C_FUNC CUresult cuFlushGPUDirectRDMAWrites(CUflushGPUDirectRDMAWritesTarget target,
                                                  CUflushGPUDirectRDMAWritesScope scope) {
    INTERCEPT_LOG("intercepted cuFlushGPUDirectRDMAWrites");
    using func_ptr = CUresult (*)(CUflushGPUDirectRDMAWritesTarget, CUflushGPUDirectRDMAWritesScope);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuFlushGPUDirectRDMAWrites"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(target, scope);
}

EXPORT_C_FUNC CUresult cuDeviceGetProperties(CUdevprop *prop, CUdevice dev) {
    INTERCEPT_LOG("intercepted cuDeviceGetProperties");
    using func_ptr = CUresult (*)(CUdevprop *, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuDeviceGetProperties"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(prop, dev);
}

EXPORT_C_FUNC CUresult cuDeviceComputeCapability(int *major, int *minor, CUdevice dev) {
    INTERCEPT_LOG("intercepted cuDeviceComputeCapability");
    using func_ptr = CUresult (*)(int *, int *, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuDeviceComputeCapability"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(major, minor, dev);
}

EXPORT_C_FUNC CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
    INTERCEPT_LOG("intercepted cuDevicePrimaryCtxRetain");
    using func_ptr = CUresult (*)(CUcontext *, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuDevicePrimaryCtxRetain"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pctx, dev);
}

EXPORT_C_FUNC CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
    INTERCEPT_LOG("intercepted cuDevicePrimaryCtxRelease");
    using func_ptr = CUresult (*)(CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuDevicePrimaryCtxRelease"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dev);
}

// manually add
EXPORT_C_FUNC CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev) {
    INTERCEPT_LOG("intercepted cuDevicePrimaryCtxRelease_v2");
    using func_ptr = CUresult (*)(CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuDevicePrimaryCtxRelease_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dev);
}

EXPORT_C_FUNC CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags) {
    INTERCEPT_LOG("intercepted cuDevicePrimaryCtxSetFlags");
    using func_ptr = CUresult (*)(CUdevice, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuDevicePrimaryCtxSetFlags"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dev, flags);
}

// manually add
EXPORT_C_FUNC CUresult cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags) {
    INTERCEPT_LOG("intercepted cuDevicePrimaryCtxSetFlags_v2");
    using func_ptr = CUresult (*)(CUdevice, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuDevicePrimaryCtxSetFlags_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dev, flags);
}

EXPORT_C_FUNC CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active) {
    INTERCEPT_LOG("intercepted cuDevicePrimaryCtxGetState");
    using func_ptr = CUresult (*)(CUdevice, unsigned int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuDevicePrimaryCtxGetState"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dev, flags, active);
}

EXPORT_C_FUNC CUresult cuDevicePrimaryCtxReset(CUdevice dev) {
    INTERCEPT_LOG("intercepted cuDevicePrimaryCtxReset");
    using func_ptr = CUresult (*)(CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuDevicePrimaryCtxReset"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dev);
}

// manually add
EXPORT_C_FUNC CUresult cuDevicePrimaryCtxReset_v2(CUdevice dev) {
    INTERCEPT_LOG("intercepted cuDevicePrimaryCtxReset_v2");
    using func_ptr = CUresult (*)(CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuDevicePrimaryCtxReset_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dev);
}

EXPORT_C_FUNC CUresult cuDeviceGetExecAffinitySupport(int *pi, CUexecAffinityType type, CUdevice dev) {
    INTERCEPT_LOG("intercepted cuDeviceGetExecAffinitySupport");
    using func_ptr = CUresult (*)(int *, CUexecAffinityType, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuDeviceGetExecAffinitySupport"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pi, type, dev);
}

EXPORT_C_FUNC CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) {
    INTERCEPT_LOG("intercepted cuCtxCreate");
    using func_ptr = CUresult (*)(CUcontext *, unsigned int, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuCtxCreate"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pctx, flags, dev);
}

// manually add
EXPORT_C_FUNC CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev) {
    INTERCEPT_LOG("intercepted cuCtxCreate_v2");
    using func_ptr = CUresult (*)(CUcontext *, unsigned int, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuCtxCreate_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pctx, flags, dev);
}

EXPORT_C_FUNC CUresult cuCtxCreate_v3(CUcontext *pctx, CUexecAffinityParam *paramsArray, int numParams,
                                      unsigned int flags, CUdevice dev) {
    INTERCEPT_LOG("intercepted cuCtxCreate_v3");
    using func_ptr = CUresult (*)(CUcontext *, CUexecAffinityParam *, int, unsigned int, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuCtxCreate_v3"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pctx, paramsArray, numParams, flags, dev);
}

EXPORT_C_FUNC CUresult cuCtxDestroy(CUcontext ctx) {
    INTERCEPT_LOG("intercepted cuCtxDestroy");
    using func_ptr = CUresult (*)(CUcontext);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuCtxDestroy"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(ctx);
}

// manually add
EXPORT_C_FUNC CUresult cuCtxDestroy_v2(CUcontext ctx) {
    INTERCEPT_LOG("intercepted cuCtxDestroy_v2");
    using func_ptr = CUresult (*)(CUcontext);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuCtxDestroy_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(ctx);
}

EXPORT_C_FUNC CUresult cuCtxPushCurrent(CUcontext ctx) {
    INTERCEPT_LOG("intercepted cuCtxPushCurrent");
    using func_ptr = CUresult (*)(CUcontext);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuCtxPushCurrent"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(ctx);
}

// manually add
EXPORT_C_FUNC CUresult cuCtxPushCurrent_v2(CUcontext ctx) {
    INTERCEPT_LOG("intercepted cuCtxPushCurrent_v2");
    using func_ptr = CUresult (*)(CUcontext);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuCtxPushCurrent_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(ctx);
}

EXPORT_C_FUNC CUresult cuCtxPopCurrent(CUcontext *pctx) {
    INTERCEPT_LOG("intercepted cuCtxPopCurrent");
    using func_ptr = CUresult (*)(CUcontext *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuCtxPopCurrent"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pctx);
}

// manually add
EXPORT_C_FUNC CUresult cuCtxPopCurrent_v2(CUcontext *pctx) {
    INTERCEPT_LOG("intercepted cuCtxPopCurrent_v2");
    using func_ptr = CUresult (*)(CUcontext *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuCtxPopCurrent_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pctx);
}

EXPORT_C_FUNC CUresult cuCtxSetCurrent(CUcontext ctx) {
    INTERCEPT_LOG("intercepted cuCtxSetCurrent");
    using func_ptr = CUresult (*)(CUcontext);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuCtxSetCurrent"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(ctx);
}

EXPORT_C_FUNC CUresult cuCtxGetCurrent(CUcontext *pctx) {
    INTERCEPT_LOG("intercepted cuCtxGetCurrent");
    using func_ptr = CUresult (*)(CUcontext *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuCtxGetCurrent"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pctx);
}

EXPORT_C_FUNC CUresult cuCtxGetDevice(CUdevice *device) {
    INTERCEPT_LOG("intercepted cuCtxGetDevice");
    using func_ptr = CUresult (*)(CUdevice *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuCtxGetDevice"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(device);
}

EXPORT_C_FUNC CUresult cuCtxGetFlags(unsigned int *flags) {
    INTERCEPT_LOG("intercepted cuCtxGetFlags");
    using func_ptr = CUresult (*)(unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuCtxGetFlags"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(flags);
}

EXPORT_C_FUNC CUresult cuCtxSynchronize() {
    INTERCEPT_LOG("intercepted cuCtxSynchronize");
    return xsched::shim::cuda::XCtxSynchronize();
}

EXPORT_C_FUNC CUresult cuCtxSetLimit(CUlimit limit, size_t value) {
    INTERCEPT_LOG("intercepted cuCtxSetLimit");
    using func_ptr = CUresult (*)(CUlimit, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuCtxSetLimit"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(limit, value);
}

EXPORT_C_FUNC CUresult cuCtxGetLimit(size_t *pvalue, CUlimit limit) {
    INTERCEPT_LOG("intercepted cuCtxGetLimit");
    using func_ptr = CUresult (*)(size_t *, CUlimit);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuCtxGetLimit"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pvalue, limit);
}

EXPORT_C_FUNC CUresult cuCtxGetCacheConfig(CUfunc_cache *pconfig) {
    INTERCEPT_LOG("intercepted cuCtxGetCacheConfig");
    using func_ptr = CUresult (*)(CUfunc_cache *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuCtxGetCacheConfig"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pconfig);
}

EXPORT_C_FUNC CUresult cuCtxSetCacheConfig(CUfunc_cache config) {
    INTERCEPT_LOG("intercepted cuCtxSetCacheConfig");
    using func_ptr = CUresult (*)(CUfunc_cache);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuCtxSetCacheConfig"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(config);
}

EXPORT_C_FUNC CUresult cuCtxGetSharedMemConfig(CUsharedconfig *pConfig) {
    INTERCEPT_LOG("intercepted cuCtxGetSharedMemConfig");
    using func_ptr = CUresult (*)(CUsharedconfig *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuCtxGetSharedMemConfig"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pConfig);
}

EXPORT_C_FUNC CUresult cuCtxSetSharedMemConfig(CUsharedconfig config) {
    INTERCEPT_LOG("intercepted cuCtxSetSharedMemConfig");
    using func_ptr = CUresult (*)(CUsharedconfig);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuCtxSetSharedMemConfig"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(config);
}

EXPORT_C_FUNC CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int *version) {
    INTERCEPT_LOG("intercepted cuCtxGetApiVersion");
    using func_ptr = CUresult (*)(CUcontext, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuCtxGetApiVersion"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(ctx, version);
}

EXPORT_C_FUNC CUresult cuCtxGetStreamPriorityRange(int *leastPriority, int *greatestPriority) {
    INTERCEPT_LOG("intercepted cuCtxGetStreamPriorityRange");
    using func_ptr = CUresult (*)(int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuCtxGetStreamPriorityRange"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(leastPriority, greatestPriority);
}

EXPORT_C_FUNC CUresult cuCtxResetPersistingL2Cache() {
    INTERCEPT_LOG("intercepted cuCtxResetPersistingL2Cache");
    using func_ptr = CUresult (*)();
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuCtxResetPersistingL2Cache"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry();
}

EXPORT_C_FUNC CUresult cuCtxGetExecAffinity(CUexecAffinityParam *pExecAffinity, CUexecAffinityType type) {
    INTERCEPT_LOG("intercepted cuCtxGetExecAffinity");
    using func_ptr = CUresult (*)(CUexecAffinityParam *, CUexecAffinityType);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuCtxGetExecAffinity"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pExecAffinity, type);
}

EXPORT_C_FUNC CUresult cuCtxAttach(CUcontext *pctx, unsigned int flags) {
    INTERCEPT_LOG("intercepted cuCtxAttach");
    using func_ptr = CUresult (*)(CUcontext *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuCtxAttach"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pctx, flags);
}

EXPORT_C_FUNC CUresult cuCtxDetach(CUcontext ctx) {
    INTERCEPT_LOG("intercepted cuCtxDetach");
    using func_ptr = CUresult (*)(CUcontext);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuCtxDetach"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(ctx);
}

EXPORT_C_FUNC CUresult cuModuleLoad(CUmodule *module, const char *fname) {
    INTERCEPT_LOG("intercepted cuModuleLoad");
    using func_ptr = CUresult (*)(CUmodule *, const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuModuleLoad"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(module, fname);
}

EXPORT_C_FUNC CUresult cuModuleLoadData(CUmodule *module, const void *image) {
    INTERCEPT_LOG("intercepted cuModuleLoadData");
    using func_ptr = CUresult (*)(CUmodule *, const void *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuModuleLoadData"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(module, image);
}

EXPORT_C_FUNC CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions,
                                          CUjit_option *options, void **optionValues) {
    INTERCEPT_LOG("intercepted cuModuleLoadDataEx");
    using func_ptr = CUresult (*)(CUmodule *, const void *, unsigned int, CUjit_option *, void **);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuModuleLoadDataEx"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(module, image, numOptions, options, optionValues);
}

EXPORT_C_FUNC CUresult cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin) {
    INTERCEPT_LOG("intercepted cuModuleLoadFatBinary");
    using func_ptr = CUresult (*)(CUmodule *, const void *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuModuleLoadFatBinary"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(module, fatCubin);
}

EXPORT_C_FUNC CUresult cuModuleUnload(CUmodule hmod) {
    INTERCEPT_LOG("intercepted cuModuleUnload");
    using func_ptr = CUresult (*)(CUmodule);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuModuleUnload"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hmod);
}

EXPORT_C_FUNC CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
    INTERCEPT_LOG("intercepted cuModuleGetFunction");
    using func_ptr = CUresult (*)(CUfunction *, CUmodule, const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuModuleGetFunction"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hfunc, hmod, name);
}

EXPORT_C_FUNC CUresult cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod,
                                         const char *name) {
    INTERCEPT_LOG("intercepted cuModuleGetGlobal");
    using func_ptr = CUresult (*)(CUdeviceptr *, size_t *, CUmodule, const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuModuleGetGlobal"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dptr, bytes, hmod, name);
}

// manually add
EXPORT_C_FUNC CUresult cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod,
                                            const char *name) {
    INTERCEPT_LOG("intercepted cuModuleGetGlobal_v2");
    using func_ptr = CUresult (*)(CUdeviceptr *, size_t *, CUmodule, const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuModuleGetGlobal_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dptr, bytes, hmod, name);
}

EXPORT_C_FUNC CUresult cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, const char *name) {
    INTERCEPT_LOG("intercepted cuModuleGetTexRef");
    using func_ptr = CUresult (*)(CUtexref *, CUmodule, const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuModuleGetTexRef"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pTexRef, hmod, name);
}

EXPORT_C_FUNC CUresult cuModuleGetSurfRef(CUsurfref *pSurfRef, CUmodule hmod, const char *name) {
    INTERCEPT_LOG("intercepted cuModuleGetSurfRef");
    using func_ptr = CUresult (*)(CUsurfref *, CUmodule, const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuModuleGetSurfRef"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pSurfRef, hmod, name);
}

EXPORT_C_FUNC CUresult cuLinkCreate(unsigned int numOptions, CUjit_option *options, void **optionValues,
                                    CUlinkState *stateOut) {
    INTERCEPT_LOG("intercepted cuLinkCreate");
    using func_ptr = CUresult (*)(unsigned int, CUjit_option *, void **, CUlinkState *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuLinkCreate"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(numOptions, options, optionValues, stateOut);
}

// manually add
EXPORT_C_FUNC CUresult cuLinkCreate_v2(unsigned int numOptions, CUjit_option *options,
                                       void **optionValues, CUlinkState *stateOut) {
    INTERCEPT_LOG("intercepted cuLinkCreate_v2");
    using func_ptr = CUresult (*)(unsigned int, CUjit_option *, void **, CUlinkState *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuLinkCreate_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(numOptions, options, optionValues, stateOut);
}

EXPORT_C_FUNC CUresult cuLinkAddData(CUlinkState state, CUjitInputType type, void *data, size_t size,
                                     const char *name, unsigned int numOptions, CUjit_option *options,
                                     void **optionValues) {
    INTERCEPT_LOG("intercepted cuLinkAddData");
    using func_ptr =
        CUresult (*)(CUlinkState, CUjitInputType, void *, size_t, const char *, unsigned int, CUjit_option *, void **);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuLinkAddData"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(state, type, data, size, name, numOptions, options, optionValues);
}

// manually add
EXPORT_C_FUNC CUresult cuLinkAddData_v2(CUlinkState state, CUjitInputType type, void *data, size_t size,
                                        const char *name, unsigned int numOptions, CUjit_option *options,
                                        void **optionValues) {
    INTERCEPT_LOG("intercepted cuLinkAddData_v2");
    using func_ptr =
        CUresult (*)(CUlinkState, CUjitInputType, void *, size_t, const char *, unsigned int, CUjit_option *, void **);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuLinkAddData_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(state, type, data, size, name, numOptions, options, optionValues);
}

EXPORT_C_FUNC CUresult cuLinkAddFile(CUlinkState state, CUjitInputType type, const char *path,
                                     unsigned int numOptions, CUjit_option *options,
                                     void **optionValues) {
    INTERCEPT_LOG("intercepted cuLinkAddFile");
    using func_ptr = CUresult (*)(CUlinkState, CUjitInputType, const char *, unsigned int, CUjit_option *, void **);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuLinkAddFile"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(state, type, path, numOptions, options, optionValues);
}

// manually add
EXPORT_C_FUNC CUresult cuLinkAddFile_v2(CUlinkState state, CUjitInputType type, const char *path,
                                        unsigned int numOptions, CUjit_option *options,
                                        void **optionValues) {
    INTERCEPT_LOG("intercepted cuLinkAddFile_v2");
    using func_ptr = CUresult (*)(CUlinkState, CUjitInputType, const char *, unsigned int, CUjit_option *, void **);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuLinkAddFile_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(state, type, path, numOptions, options, optionValues);
}

EXPORT_C_FUNC CUresult cuLinkComplete(CUlinkState state, void **cubinOut, size_t *sizeOut) {
    INTERCEPT_LOG("intercepted cuLinkComplete");
    using func_ptr = CUresult (*)(CUlinkState, void **, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuLinkComplete"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(state, cubinOut, sizeOut);
}

EXPORT_C_FUNC CUresult cuLinkDestroy(CUlinkState state) {
    INTERCEPT_LOG("intercepted cuLinkDestroy");
    using func_ptr = CUresult (*)(CUlinkState);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuLinkDestroy"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(state);
}

EXPORT_C_FUNC CUresult cuMemGetInfo(size_t *free, size_t *total) {
    INTERCEPT_LOG("intercepted cuMemGetInfo");
    using func_ptr = CUresult (*)(size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemGetInfo"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(free, total);
}

// manually add
EXPORT_C_FUNC CUresult cuMemGetInfo_v2(size_t *free, size_t *total) {
    INTERCEPT_LOG("intercepted cuMemGetInfo_v2");
    using func_ptr = CUresult (*)(size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemGetInfo_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(free, total);
}

EXPORT_C_FUNC CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
    INTERCEPT_LOG("intercepted cuMemAlloc");
    using func_ptr = CUresult (*)(CUdeviceptr *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemAlloc"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dptr, bytesize);
}

// manually add
EXPORT_C_FUNC CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
    INTERCEPT_LOG("intercepted cuMemAlloc_v2");
    using func_ptr = CUresult (*)(CUdeviceptr *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemAlloc_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dptr, bytesize);
}

EXPORT_C_FUNC CUresult cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes,
                                       size_t Height, unsigned int ElementSizeBytes) {
    INTERCEPT_LOG("intercepted cuMemAllocPitch");
    using func_ptr = CUresult (*)(CUdeviceptr *, size_t *, size_t, size_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemAllocPitch"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
}

// manually add
EXPORT_C_FUNC CUresult cuMemAllocPitch_v2(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes,
                                          size_t Height, unsigned int ElementSizeBytes) {
    INTERCEPT_LOG("intercepted cuMemAllocPitch_v2");
    using func_ptr = CUresult (*)(CUdeviceptr *, size_t *, size_t, size_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemAllocPitch_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
}

EXPORT_C_FUNC CUresult cuMemFree(CUdeviceptr dptr) {
    INTERCEPT_LOG("intercepted cuMemFree");
    using func_ptr = CUresult (*)(CUdeviceptr);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemFree"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dptr);
}

// manually add
EXPORT_C_FUNC CUresult cuMemFree_v2(CUdeviceptr dptr) {
    INTERCEPT_LOG("intercepted cuMemFree_v2");
    using func_ptr = CUresult (*)(CUdeviceptr);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemFree_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dptr);
}

EXPORT_C_FUNC CUresult cuMemGetAddressRange(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr) {
    INTERCEPT_LOG("intercepted cuMemGetAddressRange");
    using func_ptr = CUresult (*)(CUdeviceptr *, size_t *, CUdeviceptr);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemGetAddressRange"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pbase, psize, dptr);
}

// manually add
EXPORT_C_FUNC CUresult cuMemGetAddressRange_v2(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr) {
    INTERCEPT_LOG("intercepted cuMemGetAddressRange_v2");
    using func_ptr = CUresult (*)(CUdeviceptr *, size_t *, CUdeviceptr);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemGetAddressRange_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pbase, psize, dptr);
}

EXPORT_C_FUNC CUresult cuMemAllocHost(void **pp, size_t bytesize) {
    INTERCEPT_LOG("intercepted cuMemAllocHost");
    using func_ptr = CUresult (*)(void **, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemAllocHost"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pp, bytesize);
}

// manually add
EXPORT_C_FUNC CUresult cuMemAllocHost_v2(void **pp, size_t bytesize) {
    INTERCEPT_LOG("intercepted cuMemAllocHost_v2");
    using func_ptr = CUresult (*)(void **, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemAllocHost_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pp, bytesize);
}

EXPORT_C_FUNC CUresult cuMemFreeHost(void *p) {
    INTERCEPT_LOG("intercepted cuMemFreeHost");
    using func_ptr = CUresult (*)(void *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemFreeHost"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(p);
}

EXPORT_C_FUNC CUresult cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags) {
    INTERCEPT_LOG("intercepted cuMemHostAlloc");
    using func_ptr = CUresult (*)(void **, size_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemHostAlloc"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pp, bytesize, Flags);
}

EXPORT_C_FUNC CUresult cuMemHostGetDevicePointer(CUdeviceptr *pdptr, void *p, unsigned int Flags) {
    INTERCEPT_LOG("intercepted cuMemHostGetDevicePointer");
    using func_ptr = CUresult (*)(CUdeviceptr *, void *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemHostGetDevicePointer"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pdptr, p, Flags);
}

// manually add
EXPORT_C_FUNC CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr *pdptr, void *p, unsigned int Flags) {
    INTERCEPT_LOG("intercepted cuMemHostGetDevicePointer_v2");
    using func_ptr = CUresult (*)(CUdeviceptr *, void *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemHostGetDevicePointer_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pdptr, p, Flags);
}

EXPORT_C_FUNC CUresult cuMemHostGetFlags(unsigned int *pFlags, void *p) {
    INTERCEPT_LOG("intercepted cuMemHostGetFlags");
    using func_ptr = CUresult (*)(unsigned int *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemHostGetFlags"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pFlags, p);
}

EXPORT_C_FUNC CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize, unsigned int flags) {
    INTERCEPT_LOG("intercepted cuMemAllocManaged");
    using func_ptr = CUresult (*)(CUdeviceptr *, size_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemAllocManaged"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dptr, bytesize, flags);
}

EXPORT_C_FUNC CUresult cuDeviceGetByPCIBusId(CUdevice *dev, const char *pciBusId) {
    INTERCEPT_LOG("intercepted cuDeviceGetByPCIBusId");
    using func_ptr = CUresult (*)(CUdevice *, const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuDeviceGetByPCIBusId"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dev, pciBusId);
}

EXPORT_C_FUNC CUresult cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev) {
    INTERCEPT_LOG("intercepted cuDeviceGetPCIBusId");
    using func_ptr = CUresult (*)(char *, int, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuDeviceGetPCIBusId"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pciBusId, len, dev);
}

EXPORT_C_FUNC CUresult cuIpcGetEventHandle(CUipcEventHandle *pHandle, CUevent event) {
    INTERCEPT_LOG("intercepted cuIpcGetEventHandle");
    using func_ptr = CUresult (*)(CUipcEventHandle *, CUevent);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuIpcGetEventHandle"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pHandle, event);
}

EXPORT_C_FUNC CUresult cuIpcOpenEventHandle(CUevent *phEvent, CUipcEventHandle handle) {
    INTERCEPT_LOG("intercepted cuIpcOpenEventHandle");
    using func_ptr = CUresult (*)(CUevent *, CUipcEventHandle);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuIpcOpenEventHandle"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(phEvent, handle);
}

EXPORT_C_FUNC CUresult cuIpcGetMemHandle(CUipcMemHandle *pHandle, CUdeviceptr dptr) {
    INTERCEPT_LOG("intercepted cuIpcGetMemHandle");
    using func_ptr = CUresult (*)(CUipcMemHandle *, CUdeviceptr);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuIpcGetMemHandle"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pHandle, dptr);
}

EXPORT_C_FUNC CUresult cuIpcOpenMemHandle(CUdeviceptr *pdptr, CUipcMemHandle handle, unsigned int Flags) {
    INTERCEPT_LOG("intercepted cuIpcOpenMemHandle");
    using func_ptr = CUresult (*)(CUdeviceptr *, CUipcMemHandle, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuIpcOpenMemHandle"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pdptr, handle, Flags);
}

// manually add
EXPORT_C_FUNC CUresult cuIpcOpenMemHandle_v2(CUdeviceptr *pdptr, CUipcMemHandle handle,
                                             unsigned int Flags) {
    INTERCEPT_LOG("intercepted cuIpcOpenMemHandle_v2");
    using func_ptr = CUresult (*)(CUdeviceptr *, CUipcMemHandle, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuIpcOpenMemHandle_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pdptr, handle, Flags);
}

EXPORT_C_FUNC CUresult cuIpcCloseMemHandle(CUdeviceptr dptr) {
    INTERCEPT_LOG("intercepted cuIpcCloseMemHandle");
    using func_ptr = CUresult (*)(CUdeviceptr);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuIpcCloseMemHandle"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dptr);
}

EXPORT_C_FUNC CUresult cuMemHostRegister(void *p, size_t bytesize, unsigned int Flags) {
    INTERCEPT_LOG("intercepted cuMemHostRegister");
    using func_ptr = CUresult (*)(void *, size_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemHostRegister"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(p, bytesize, Flags);
}

// manually add
EXPORT_C_FUNC CUresult cuMemHostRegister_v2(void *p, size_t bytesize, unsigned int Flags) {
    INTERCEPT_LOG("intercepted cuMemHostRegister_v2");
    using func_ptr = CUresult (*)(void *, size_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemHostRegister_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(p, bytesize, Flags);
}

EXPORT_C_FUNC CUresult cuMemHostUnregister(void *p) {
    INTERCEPT_LOG("intercepted cuMemHostUnregister");
    using func_ptr = CUresult (*)(void *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemHostUnregister"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(p);
}

EXPORT_C_FUNC CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
    INTERCEPT_LOG("intercepted cuMemcpy");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(CUdeviceptr, CUdeviceptr, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemcpy"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dst, src, ByteCount);
}

EXPORT_C_FUNC CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice,
                                    CUcontext srcContext, size_t ByteCount) {
    INTERCEPT_LOG("intercepted cuMemcpyPeer");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemcpyPeer"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dstDevice, dstContext, srcDevice, srcContext, ByteCount);
}

EXPORT_C_FUNC CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
    INTERCEPT_LOG("intercepted cuMemcpyHtoD");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(CUdeviceptr, const void *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemcpyHtoD"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dstDevice, srcHost, ByteCount);
}

EXPORT_C_FUNC CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    INTERCEPT_LOG("intercepted cuMemcpyDtoH");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(void *, CUdeviceptr, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemcpyDtoH"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dstHost, srcDevice, ByteCount);
}

EXPORT_C_FUNC CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
    INTERCEPT_LOG("intercepted cuMemcpyDtoD");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(CUdeviceptr, CUdeviceptr, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemcpyDtoD"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dstDevice, srcDevice, ByteCount);
}

EXPORT_C_FUNC CUresult cuMemcpyDtoA(CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice,
                                    size_t ByteCount) {
    INTERCEPT_LOG("intercepted cuMemcpyDtoA");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(CUarray, size_t, CUdeviceptr, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemcpyDtoA"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dstArray, dstOffset, srcDevice, ByteCount);
}

EXPORT_C_FUNC CUresult cuMemcpyAtoD(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset,
                                    size_t ByteCount) {
    INTERCEPT_LOG("intercepted cuMemcpyAtoD");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(CUdeviceptr, CUarray, size_t, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemcpyAtoD"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dstDevice, srcArray, srcOffset, ByteCount);
}

EXPORT_C_FUNC CUresult cuMemcpyHtoA(CUarray dstArray, size_t dstOffset, const void *srcHost,
                                    size_t ByteCount) {
    INTERCEPT_LOG("intercepted cuMemcpyHtoA");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(CUarray, size_t, const void *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemcpyHtoA"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dstArray, dstOffset, srcHost, ByteCount);
}

EXPORT_C_FUNC CUresult cuMemcpyAtoH(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount) {
    INTERCEPT_LOG("intercepted cuMemcpyAtoH");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(void *, CUarray, size_t, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemcpyAtoH"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dstHost, srcArray, srcOffset, ByteCount);
}

EXPORT_C_FUNC CUresult cuMemcpyAtoA(CUarray dstArray, size_t dstOffset, CUarray srcArray,
                                    size_t srcOffset, size_t ByteCount) {
    INTERCEPT_LOG("intercepted cuMemcpyAtoA");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(CUarray, size_t, CUarray, size_t, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemcpyAtoA"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dstArray, dstOffset, srcArray, srcOffset, ByteCount);
}

EXPORT_C_FUNC CUresult cuMemcpy2D(const CUDA_MEMCPY2D *pCopy) {
    INTERCEPT_LOG("intercepted cuMemcpy2D");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(const CUDA_MEMCPY2D *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemcpy2D"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pCopy);
}

EXPORT_C_FUNC CUresult cuMemcpy2DUnaligned(const CUDA_MEMCPY2D *pCopy) {
    INTERCEPT_LOG("intercepted cuMemcpy2DUnaligned");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(const CUDA_MEMCPY2D *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemcpy2DUnaligned"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pCopy);
}

EXPORT_C_FUNC CUresult cuMemcpy3D(const CUDA_MEMCPY3D *pCopy) {
    INTERCEPT_LOG("intercepted cuMemcpy3D");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(const CUDA_MEMCPY3D *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemcpy3D"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pCopy);
}

EXPORT_C_FUNC CUresult cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER *pCopy) {
    INTERCEPT_LOG("intercepted cuMemcpy3DPeer");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(const CUDA_MEMCPY3D_PEER *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemcpy3DPeer"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pCopy);
}

EXPORT_C_FUNC CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount,
                                     CUstream hStream) {
    INTERCEPT_LOG("intercepted cuMemcpyAsync");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemcpyAsync"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dst, src, ByteCount, hStream);
}

EXPORT_C_FUNC CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext,
                                         CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount,
                                         CUstream hStream) {
    INTERCEPT_LOG("intercepted cuMemcpyPeerAsync");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemcpyPeerAsync"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream);
}

EXPORT_C_FUNC CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount,
                                         CUstream hStream) {
    INTERCEPT_LOG("intercepted cuMemcpyHtoDAsync");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(CUdeviceptr, const void *, size_t, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemcpyHtoDAsync"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dstDevice, srcHost, ByteCount, hStream);
}

EXPORT_C_FUNC CUresult cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount,
                                         CUstream hStream) {
    INTERCEPT_LOG("intercepted cuMemcpyDtoHAsync");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(void *, CUdeviceptr, size_t, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemcpyDtoHAsync"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dstHost, srcDevice, ByteCount, hStream);
}

EXPORT_C_FUNC CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount,
                                         CUstream hStream) {
    INTERCEPT_LOG("intercepted cuMemcpyDtoDAsync");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemcpyDtoDAsync"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dstDevice, srcDevice, ByteCount, hStream);
}

EXPORT_C_FUNC CUresult cuMemcpyHtoAAsync(CUarray dstArray, size_t dstOffset, const void *srcHost,
                                         size_t ByteCount, CUstream hStream) {
    INTERCEPT_LOG("intercepted cuMemcpyHtoAAsync");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(CUarray, size_t, const void *, size_t, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemcpyHtoAAsync"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dstArray, dstOffset, srcHost, ByteCount, hStream);
}

EXPORT_C_FUNC CUresult cuMemcpyAtoHAsync(void *dstHost, CUarray srcArray, size_t srcOffset,
                                         size_t ByteCount, CUstream hStream) {
    INTERCEPT_LOG("intercepted cuMemcpyAtoHAsync");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(void *, CUarray, size_t, size_t, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemcpyAtoHAsync"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dstHost, srcArray, srcOffset, ByteCount, hStream);
}

EXPORT_C_FUNC CUresult cuMemcpy2DAsync(const CUDA_MEMCPY2D *pCopy, CUstream hStream) {
    INTERCEPT_LOG("intercepted cuMemcpy2DAsync");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(const CUDA_MEMCPY2D *, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemcpy2DAsync"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pCopy, hStream);
}

EXPORT_C_FUNC CUresult cuMemcpy3DAsync(const CUDA_MEMCPY3D *pCopy, CUstream hStream) {
    INTERCEPT_LOG("intercepted cuMemcpy3DAsync");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(const CUDA_MEMCPY3D *, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemcpy3DAsync"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pCopy, hStream);
}

EXPORT_C_FUNC CUresult cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER *pCopy, CUstream hStream) {
    INTERCEPT_LOG("intercepted cuMemcpy3DPeerAsync");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(const CUDA_MEMCPY3D_PEER *, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemcpy3DPeerAsync"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pCopy, hStream);
}

EXPORT_C_FUNC CUresult cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
    INTERCEPT_LOG("intercepted cuMemsetD8");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(CUdeviceptr, unsigned char, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemsetD8"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dstDevice, uc, N);
}

EXPORT_C_FUNC CUresult cuMemsetD16(CUdeviceptr dstDevice, unsigned short us, size_t N) {
    INTERCEPT_LOG("intercepted cuMemsetD16");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(CUdeviceptr, unsigned short, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemsetD16"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dstDevice, us, N);
}

EXPORT_C_FUNC CUresult cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
    INTERCEPT_LOG("intercepted cuMemsetD32");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(CUdeviceptr, unsigned int, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemsetD32"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dstDevice, ui, N);
}

EXPORT_C_FUNC CUresult cuMemsetD2D8(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc,
                                    size_t Width, size_t Height) {
    INTERCEPT_LOG("intercepted cuMemsetD2D8");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(CUdeviceptr, size_t, unsigned char, size_t, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemsetD2D8"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dstDevice, dstPitch, uc, Width, Height);
}

EXPORT_C_FUNC CUresult cuMemsetD2D16(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us,
                                     size_t Width, size_t Height) {
    INTERCEPT_LOG("intercepted cuMemsetD2D16");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(CUdeviceptr, size_t, unsigned short, size_t, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemsetD2D16"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dstDevice, dstPitch, us, Width, Height);
}

EXPORT_C_FUNC CUresult cuMemsetD2D32(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui,
                                     size_t Width, size_t Height) {
    INTERCEPT_LOG("intercepted cuMemsetD2D32");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(CUdeviceptr, size_t, unsigned int, size_t, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemsetD2D32"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dstDevice, dstPitch, ui, Width, Height);
}

EXPORT_C_FUNC CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N,
                                       CUstream hStream) {
    INTERCEPT_LOG("intercepted cuMemsetD8Async ptr(0x%llx), val(0x%x), n(%lu), stream(%p)", dstDevice, uc, N, hStream);
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    return xsched::shim::cuda::XMemsetD8Async(dstDevice, uc, N, hStream);
}

EXPORT_C_FUNC CUresult cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N,
                                        CUstream hStream) {
    INTERCEPT_LOG("intercepted cuMemsetD16Async ptr(0x%llx), val(0x%x), n(%lu), stream(%p)", dstDevice, us, N, hStream);
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    return xsched::shim::cuda::XMemsetD16Async(dstDevice, us, N, hStream);
}

EXPORT_C_FUNC CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N,
                                        CUstream hStream) {
    INTERCEPT_LOG("intercepted cuMemsetD32Async ptr(0x%llx), val(0x%x), n(%lu), stream(%p)", dstDevice, ui, N, hStream);
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    return xsched::shim::cuda::XMemsetD16Async(dstDevice, ui, N, hStream);
}

EXPORT_C_FUNC CUresult cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc,
                                         size_t Width, size_t Height, CUstream hStream) {
    INTERCEPT_LOG("intercepted cuMemsetD2D8Async ptr(0x%llx), pitch(%lu), val(0x%x), width(%lu), height(%lu), stream(%p)", dstDevice, dstPitch, uc, Width, Height, hStream);
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    return xsched::shim::cuda::XMemsetD2D8Async(dstDevice, dstPitch, uc, Width, Height, hStream);
}

EXPORT_C_FUNC CUresult cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us,
                                          size_t Width, size_t Height, CUstream hStream) {
    INTERCEPT_LOG("intercepted cuMemsetD2D16Async ptr(0x%llx), pitch(%lu), val(0x%x), width(%lu), height(%lu), stream(%p)", dstDevice, dstPitch, us, Width, Height, hStream);
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    return xsched::shim::cuda::XMemsetD2D16Async(dstDevice, dstPitch, us, Width, Height, hStream);
}

EXPORT_C_FUNC CUresult cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui,
                                          size_t Width, size_t Height, CUstream hStream) {
    INTERCEPT_LOG("intercepted cuMemsetD2D32Async ptr(0x%llx), pitch(%lu), val(0x%x), width(%lu), height(%lu), stream(%p)", dstDevice, dstPitch, ui, Width, Height, hStream);
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    return xsched::shim::cuda::XMemsetD2D32Async(dstDevice, dstPitch, ui, Width, Height, hStream);
}

EXPORT_C_FUNC CUresult cuArrayCreate(CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
    INTERCEPT_LOG("intercepted cuArrayCreate");
    using func_ptr = CUresult (*)(CUarray *, const CUDA_ARRAY_DESCRIPTOR *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuArrayCreate"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pHandle, pAllocateArray);
}

// manually add
EXPORT_C_FUNC CUresult cuArrayCreate_v2(CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
    INTERCEPT_LOG("intercepted cuArrayCreate_v2");
    using func_ptr = CUresult (*)(CUarray *, const CUDA_ARRAY_DESCRIPTOR *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuArrayCreate_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pHandle, pAllocateArray);
}

EXPORT_C_FUNC CUresult cuArrayGetDescriptor(CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray) {
    INTERCEPT_LOG("intercepted cuArrayGetDescriptor");
    using func_ptr = CUresult (*)(CUDA_ARRAY_DESCRIPTOR *, CUarray);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuArrayGetDescriptor"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pArrayDescriptor, hArray);
}

// manually add
EXPORT_C_FUNC CUresult cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray) {
    INTERCEPT_LOG("intercepted cuArrayGetDescriptor_v2");
    using func_ptr = CUresult (*)(CUDA_ARRAY_DESCRIPTOR *, CUarray);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuArrayGetDescriptor_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pArrayDescriptor, hArray);
}

EXPORT_C_FUNC CUresult cuArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES *sparseProperties,
                                                  CUarray array) {
    INTERCEPT_LOG("intercepted cuArrayGetSparseProperties");
    using func_ptr = CUresult (*)(CUDA_ARRAY_SPARSE_PROPERTIES *, CUarray);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuArrayGetSparseProperties"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(sparseProperties, array);
}

EXPORT_C_FUNC CUresult cuMipmappedArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES *sparseProperties,
                                                           CUmipmappedArray mipmap) {
    INTERCEPT_LOG("intercepted cuMipmappedArrayGetSparseProperties");
    using func_ptr = CUresult (*)(CUDA_ARRAY_SPARSE_PROPERTIES *, CUmipmappedArray);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMipmappedArrayGetSparseProperties"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(sparseProperties, mipmap);
}

EXPORT_C_FUNC CUresult cuArrayGetPlane(CUarray *pPlaneArray, CUarray hArray, unsigned int planeIdx) {
    INTERCEPT_LOG("intercepted cuArrayGetPlane");
    using func_ptr = CUresult (*)(CUarray *, CUarray, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuArrayGetPlane"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pPlaneArray, hArray, planeIdx);
}

EXPORT_C_FUNC CUresult cuArrayDestroy(CUarray hArray) {
    INTERCEPT_LOG("intercepted cuArrayDestroy");
    using func_ptr = CUresult (*)(CUarray);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuArrayDestroy"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hArray);
}

EXPORT_C_FUNC CUresult cuArray3DCreate(CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
    INTERCEPT_LOG("intercepted cuArray3DCreate");
    using func_ptr = CUresult (*)(CUarray *, const CUDA_ARRAY3D_DESCRIPTOR *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuArray3DCreate"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pHandle, pAllocateArray);
}

// manually add
EXPORT_C_FUNC CUresult cuArray3DCreate_v2(CUarray *pHandle,
                                          const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
    INTERCEPT_LOG("intercepted cuArray3DCreate_v2");
    using func_ptr = CUresult (*)(CUarray *, const CUDA_ARRAY3D_DESCRIPTOR *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuArray3DCreate_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pHandle, pAllocateArray);
}

EXPORT_C_FUNC CUresult cuArray3DGetDescriptor(CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray) {
    INTERCEPT_LOG("intercepted cuArray3DGetDescriptor");
    using func_ptr = CUresult (*)(CUDA_ARRAY3D_DESCRIPTOR *, CUarray);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuArray3DGetDescriptor"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pArrayDescriptor, hArray);
}

// manually add
EXPORT_C_FUNC CUresult cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor,
                                                 CUarray hArray) {
    INTERCEPT_LOG("intercepted cuArray3DGetDescriptor_v2");
    using func_ptr = CUresult (*)(CUDA_ARRAY3D_DESCRIPTOR *, CUarray);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuArray3DGetDescriptor_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pArrayDescriptor, hArray);
}

EXPORT_C_FUNC CUresult cuMipmappedArrayCreate(CUmipmappedArray *pHandle,
                                              const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc,
                                              unsigned int numMipmapLevels) {
    INTERCEPT_LOG("intercepted cuMipmappedArrayCreate");
    using func_ptr = CUresult (*)(CUmipmappedArray *, const CUDA_ARRAY3D_DESCRIPTOR *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMipmappedArrayCreate"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pHandle, pMipmappedArrayDesc, numMipmapLevels);
}

EXPORT_C_FUNC CUresult cuMipmappedArrayGetLevel(CUarray *pLevelArray, CUmipmappedArray hMipmappedArray,
                                                unsigned int level) {
    INTERCEPT_LOG("intercepted cuMipmappedArrayGetLevel");
    using func_ptr = CUresult (*)(CUarray *, CUmipmappedArray, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMipmappedArrayGetLevel"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pLevelArray, hMipmappedArray, level);
}

EXPORT_C_FUNC CUresult cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray) {
    INTERCEPT_LOG("intercepted cuMipmappedArrayDestroy");
    using func_ptr = CUresult (*)(CUmipmappedArray);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMipmappedArrayDestroy"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hMipmappedArray);
}

EXPORT_C_FUNC CUresult cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment,
                                           CUdeviceptr addr, unsigned long long flags) {
    INTERCEPT_LOG("intercepted cuMemAddressReserve");
    using func_ptr = CUresult (*)(CUdeviceptr *, size_t, size_t, CUdeviceptr, unsigned long long);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemAddressReserve"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(ptr, size, alignment, addr, flags);
}

EXPORT_C_FUNC CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size) {
    INTERCEPT_LOG("intercepted cuMemAddressFree");
    using func_ptr = CUresult (*)(CUdeviceptr, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemAddressFree"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(ptr, size);
}

EXPORT_C_FUNC CUresult cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size,
                                   const CUmemAllocationProp *prop, unsigned long long flags) {
    INTERCEPT_LOG("intercepted cuMemCreate");
    using func_ptr =
        CUresult (*)(CUmemGenericAllocationHandle *, size_t, const CUmemAllocationProp *, unsigned long long);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemCreate"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(handle, size, prop, flags);
}

EXPORT_C_FUNC CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
    INTERCEPT_LOG("intercepted cuMemRelease");
    using func_ptr = CUresult (*)(CUmemGenericAllocationHandle);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemRelease"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(handle);
}

EXPORT_C_FUNC CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                                CUmemGenericAllocationHandle handle, unsigned long long flags) {
    INTERCEPT_LOG("intercepted cuMemMap");
    using func_ptr = CUresult (*)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemMap"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(ptr, size, offset, handle, flags);
}

EXPORT_C_FUNC CUresult cuMemMapArrayAsync(CUarrayMapInfo *mapInfoList, unsigned int count,
                                          CUstream hStream) {
    INTERCEPT_LOG("intercepted cuMemMapArrayAsync");
    using func_ptr = CUresult (*)(CUarrayMapInfo *, unsigned int, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemMapArrayAsync"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(mapInfoList, count, hStream);
}

EXPORT_C_FUNC CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) {
    INTERCEPT_LOG("intercepted cuMemUnmap");
    using func_ptr = CUresult (*)(CUdeviceptr, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemUnmap"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(ptr, size);
}

EXPORT_C_FUNC CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc *desc,
                                      size_t count) {
    INTERCEPT_LOG("intercepted cuMemSetAccess");
    using func_ptr = CUresult (*)(CUdeviceptr, size_t, const CUmemAccessDesc *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemSetAccess"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(ptr, size, desc, count);
}

EXPORT_C_FUNC CUresult cuMemGetAccess(unsigned long long *flags, const CUmemLocation *location,
                                      CUdeviceptr ptr) {
    INTERCEPT_LOG("intercepted cuMemGetAccess");
    using func_ptr = CUresult (*)(unsigned long long *, const CUmemLocation *, CUdeviceptr);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemGetAccess"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(flags, location, ptr);
}

EXPORT_C_FUNC CUresult cuMemExportToShareableHandle(void *shareableHandle,
                                                    CUmemGenericAllocationHandle handle,
                                                    CUmemAllocationHandleType handleType,
                                                    unsigned long long flags) {
    INTERCEPT_LOG("intercepted cuMemExportToShareableHandle");
    using func_ptr = CUresult (*)(void *, CUmemGenericAllocationHandle, CUmemAllocationHandleType, unsigned long long);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemExportToShareableHandle"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(shareableHandle, handle, handleType, flags);
}

EXPORT_C_FUNC CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle *handle,
                                                      void *osHandle,
                                                      CUmemAllocationHandleType shHandleType) {
    INTERCEPT_LOG("intercepted cuMemImportFromShareableHandle");
    using func_ptr = CUresult (*)(CUmemGenericAllocationHandle *, void *, CUmemAllocationHandleType);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemImportFromShareableHandle"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(handle, osHandle, shHandleType);
}

EXPORT_C_FUNC CUresult cuMemGetAllocationGranularity(size_t *granularity, const CUmemAllocationProp *prop,
                                                     CUmemAllocationGranularity_flags option) {
    INTERCEPT_LOG("intercepted cuMemGetAllocationGranularity");
    using func_ptr = CUresult (*)(size_t *, const CUmemAllocationProp *, CUmemAllocationGranularity_flags);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemGetAllocationGranularity"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(granularity, prop, option);
}

EXPORT_C_FUNC CUresult cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp *prop,
                                                              CUmemGenericAllocationHandle handle) {
    INTERCEPT_LOG("intercepted cuMemGetAllocationPropertiesFromHandle");
    using func_ptr = CUresult (*)(CUmemAllocationProp *, CUmemGenericAllocationHandle);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemGetAllocationPropertiesFromHandle"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(prop, handle);
}

EXPORT_C_FUNC CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle *handle, void *addr) {
    INTERCEPT_LOG("intercepted cuMemRetainAllocationHandle");
    using func_ptr = CUresult (*)(CUmemGenericAllocationHandle *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemRetainAllocationHandle"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(handle, addr);
}

EXPORT_C_FUNC CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) {
    INTERCEPT_LOG("intercepted cuMemFreeAsync ptr(0x%llx), stream(%p)", dptr, hStream);
    return xsched::shim::cuda::XMemFreeAsync(dptr, hStream);
}

EXPORT_C_FUNC CUresult cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, CUstream hStream) {
    INTERCEPT_LOG("intercepted cuMemAllocAsync size(%lu), stream(%p)", bytesize, hStream);
    return xsched::shim::cuda::XMemAllocAsync(dptr, bytesize, hStream);
}

EXPORT_C_FUNC CUresult cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep) {
    INTERCEPT_LOG("intercepted cuMemPoolTrimTo");
    using func_ptr = CUresult (*)(CUmemoryPool, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemPoolTrimTo"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pool, minBytesToKeep);
}

EXPORT_C_FUNC CUresult cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void *value) {
    INTERCEPT_LOG("intercepted cuMemPoolSetAttribute");
    using func_ptr = CUresult (*)(CUmemoryPool, CUmemPool_attribute, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemPoolSetAttribute"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pool, attr, value);
}

EXPORT_C_FUNC CUresult cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void *value) {
    INTERCEPT_LOG("intercepted cuMemPoolGetAttribute");
    using func_ptr = CUresult (*)(CUmemoryPool, CUmemPool_attribute, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemPoolGetAttribute"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pool, attr, value);
}

EXPORT_C_FUNC CUresult cuMemPoolSetAccess(CUmemoryPool pool, const CUmemAccessDesc *map, size_t count) {
    INTERCEPT_LOG("intercepted cuMemPoolSetAccess");
    using func_ptr = CUresult (*)(CUmemoryPool, const CUmemAccessDesc *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemPoolSetAccess"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pool, map, count);
}

EXPORT_C_FUNC CUresult cuMemPoolGetAccess(CUmemAccess_flags *flags, CUmemoryPool memPool,
                                          CUmemLocation *location) {
    INTERCEPT_LOG("intercepted cuMemPoolGetAccess");
    using func_ptr = CUresult (*)(CUmemAccess_flags *, CUmemoryPool, CUmemLocation *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemPoolGetAccess"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(flags, memPool, location);
}

EXPORT_C_FUNC CUresult cuMemPoolCreate(CUmemoryPool *pool, const CUmemPoolProps *poolProps) {
    INTERCEPT_LOG("intercepted cuMemPoolCreate");
    using func_ptr = CUresult (*)(CUmemoryPool *, const CUmemPoolProps *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemPoolCreate"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pool, poolProps);
}

EXPORT_C_FUNC CUresult cuMemPoolDestroy(CUmemoryPool pool) {
    INTERCEPT_LOG("intercepted cuMemPoolDestroy");
    using func_ptr = CUresult (*)(CUmemoryPool);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemPoolDestroy"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pool);
}

EXPORT_C_FUNC CUresult cuMemAllocFromPoolAsync(CUdeviceptr *dptr, size_t bytesize, CUmemoryPool pool,
                                               CUstream hStream) {
    INTERCEPT_LOG("intercepted cuMemAllocFromPoolAsync");
    using func_ptr = CUresult (*)(CUdeviceptr *, size_t, CUmemoryPool, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemAllocFromPoolAsync"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dptr, bytesize, pool, hStream);
}

EXPORT_C_FUNC CUresult cuMemPoolExportToShareableHandle(void *handle_out, CUmemoryPool pool,
                                                        CUmemAllocationHandleType handleType,
                                                        unsigned long long flags) {
    INTERCEPT_LOG("intercepted cuMemPoolExportToShareableHandle");
    using func_ptr = CUresult (*)(void *, CUmemoryPool, CUmemAllocationHandleType, unsigned long long);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemPoolExportToShareableHandle"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(handle_out, pool, handleType, flags);
}

EXPORT_C_FUNC CUresult cuMemPoolImportFromShareableHandle(CUmemoryPool *pool_out, void *handle,
                                                          CUmemAllocationHandleType handleType,
                                                          unsigned long long flags) {
    INTERCEPT_LOG("intercepted cuMemPoolImportFromShareableHandle");
    using func_ptr = CUresult (*)(CUmemoryPool *, void *, CUmemAllocationHandleType, unsigned long long);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemPoolImportFromShareableHandle"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pool_out, handle, handleType, flags);
}

EXPORT_C_FUNC CUresult cuMemPoolExportPointer(CUmemPoolPtrExportData *shareData_out, CUdeviceptr ptr) {
    INTERCEPT_LOG("intercepted cuMemPoolExportPointer");
    using func_ptr = CUresult (*)(CUmemPoolPtrExportData *, CUdeviceptr);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemPoolExportPointer"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(shareData_out, ptr);
}

EXPORT_C_FUNC CUresult cuMemPoolImportPointer(CUdeviceptr *ptr_out, CUmemoryPool pool,
                                              CUmemPoolPtrExportData *shareData) {
    INTERCEPT_LOG("intercepted cuMemPoolImportPointer");
    using func_ptr = CUresult (*)(CUdeviceptr *, CUmemoryPool, CUmemPoolPtrExportData *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemPoolImportPointer"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(ptr_out, pool, shareData);
}

EXPORT_C_FUNC CUresult cuPointerGetAttribute(void *data, CUpointer_attribute attribute, CUdeviceptr ptr) {
    INTERCEPT_LOG("intercepted cuPointerGetAttribute");
    using func_ptr = CUresult (*)(void *, CUpointer_attribute, CUdeviceptr);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuPointerGetAttribute"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(data, attribute, ptr);
}

EXPORT_C_FUNC CUresult cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count, CUdevice dstDevice,
                                          CUstream hStream) {
    INTERCEPT_LOG("intercepted cuMemPrefetchAsync");
    using func_ptr = CUresult (*)(CUdeviceptr, size_t, CUdevice, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemPrefetchAsync"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(devPtr, count, dstDevice, hStream);
}

EXPORT_C_FUNC CUresult cuMemAdvise(CUdeviceptr devPtr, size_t count, CUmem_advise advice,
                                   CUdevice device) {
    INTERCEPT_LOG("intercepted cuMemAdvise");
    using func_ptr = CUresult (*)(CUdeviceptr, size_t, CUmem_advise, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemAdvise"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(devPtr, count, advice, device);
}

EXPORT_C_FUNC CUresult cuMemRangeGetAttribute(void *data, size_t dataSize,
                                              CUmem_range_attribute attribute, CUdeviceptr devPtr,
                                              size_t count) {
    INTERCEPT_LOG("intercepted cuMemRangeGetAttribute");
    using func_ptr = CUresult (*)(void *, size_t, CUmem_range_attribute, CUdeviceptr, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemRangeGetAttribute"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(data, dataSize, attribute, devPtr, count);
}

EXPORT_C_FUNC CUresult cuMemRangeGetAttributes(void **data, size_t *dataSizes,
                                               CUmem_range_attribute *attributes, size_t numAttributes,
                                               CUdeviceptr devPtr, size_t count) {
    INTERCEPT_LOG("intercepted cuMemRangeGetAttributes");
    using func_ptr = CUresult (*)(void **, size_t *, CUmem_range_attribute *, size_t, CUdeviceptr, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemRangeGetAttributes"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(data, dataSizes, attributes, numAttributes, devPtr, count);
}

EXPORT_C_FUNC CUresult cuPointerSetAttribute(const void *value, CUpointer_attribute attribute,
                                             CUdeviceptr ptr) {
    INTERCEPT_LOG("intercepted cuPointerSetAttribute");
    using func_ptr = CUresult (*)(const void *, CUpointer_attribute, CUdeviceptr);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuPointerSetAttribute"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(value, attribute, ptr);
}

EXPORT_C_FUNC CUresult cuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute *attributes,
                                              void **data, CUdeviceptr ptr) {
    INTERCEPT_LOG("intercepted cuPointerGetAttributes");
    using func_ptr = CUresult (*)(unsigned int, CUpointer_attribute *, void **, CUdeviceptr);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuPointerGetAttributes"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(numAttributes, attributes, data, ptr);
}

EXPORT_C_FUNC CUresult cuStreamCreate(CUstream *phStream, unsigned int Flags) {
    INTERCEPT_LOG("intercepted cuStreamCreate");
    return xsched::shim::cuda::XStreamCreate(phStream, Flags);
}

EXPORT_C_FUNC CUresult cuStreamCreateWithPriority(CUstream *phStream, unsigned int flags, int priority) {
    INTERCEPT_LOG("intercepted cuStreamCreateWithPriority");
    return xsched::shim::cuda::XStreamCreateWithPriority(phStream, flags, priority);
}

EXPORT_C_FUNC CUresult cuStreamGetPriority(CUstream hStream, int *priority) {
    INTERCEPT_LOG("intercepted cuStreamGetPriority");
    using func_ptr = CUresult (*)(CUstream, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuStreamGetPriority"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hStream, priority);
}

EXPORT_C_FUNC CUresult cuStreamGetFlags(CUstream hStream, unsigned int *flags) {
    INTERCEPT_LOG("intercepted cuStreamGetFlags");
    using func_ptr = CUresult (*)(CUstream, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuStreamGetFlags"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hStream, flags);
}

EXPORT_C_FUNC CUresult cuStreamGetCtx(CUstream hStream, CUcontext *pctx) {
    INTERCEPT_LOG("intercepted cuStreamGetCtx");
    using func_ptr = CUresult (*)(CUstream, CUcontext *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuStreamGetCtx"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hStream, pctx);
}

EXPORT_C_FUNC CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) {
    INTERCEPT_LOG("intercepted cuStreamWaitEvent stream(%p), event(%p), flags(0x%x)", hStream, hEvent, Flags);
    return xsched::shim::cuda::XStreamWaitEvent(hStream, hEvent, Flags);
}

EXPORT_C_FUNC CUresult cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, void *userData,
                                           unsigned int flags) {
    INTERCEPT_LOG("intercepted cuStreamAddCallback");
    using func_ptr = CUresult (*)(CUstream, CUstreamCallback, void *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuStreamAddCallback"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hStream, callback, userData, flags);
}

EXPORT_C_FUNC CUresult cuStreamBeginCapture(CUstream hStream, CUstreamCaptureMode mode) {
    INTERCEPT_LOG("intercepted cuStreamBeginCapture");
    using func_ptr = CUresult (*)(CUstream, CUstreamCaptureMode);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuStreamBeginCapture"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hStream, mode);
}

EXPORT_C_FUNC CUresult cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode *mode) {
    INTERCEPT_LOG("intercepted cuThreadExchangeStreamCaptureMode");
    using func_ptr = CUresult (*)(CUstreamCaptureMode *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuThreadExchangeStreamCaptureMode"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(mode);
}

EXPORT_C_FUNC CUresult cuStreamEndCapture(CUstream hStream, CUgraph *phGraph) {
    INTERCEPT_LOG("intercepted cuStreamEndCapture");
    using func_ptr = CUresult (*)(CUstream, CUgraph *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuStreamEndCapture"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hStream, phGraph);
}

EXPORT_C_FUNC CUresult cuStreamIsCapturing(CUstream hStream, CUstreamCaptureStatus *captureStatus) {
    INTERCEPT_LOG("intercepted cuStreamIsCapturing");
    using func_ptr = CUresult (*)(CUstream, CUstreamCaptureStatus *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuStreamIsCapturing"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hStream, captureStatus);
}

EXPORT_C_FUNC CUresult cuStreamGetCaptureInfo(CUstream hStream, CUstreamCaptureStatus *captureStatus_out,
                                              cuuint64_t *id_out) {
    INTERCEPT_LOG("intercepted cuStreamGetCaptureInfo");
    using func_ptr = CUresult (*)(CUstream, CUstreamCaptureStatus *, cuuint64_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuStreamGetCaptureInfo"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hStream, captureStatus_out, id_out);
}

EXPORT_C_FUNC CUresult cuStreamGetCaptureInfo_v2(CUstream hStream,
                                                 CUstreamCaptureStatus *captureStatus_out,
                                                 cuuint64_t *id_out, CUgraph *graph_out,
                                                 const CUgraphNode **dependencies_out,
                                                 size_t *numDependencies_out) {
    INTERCEPT_LOG("intercepted cuStreamGetCaptureInfo_v2");
    using func_ptr =
        CUresult (*)(CUstream, CUstreamCaptureStatus *, cuuint64_t *, CUgraph *, const CUgraphNode **, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuStreamGetCaptureInfo_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hStream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out);
}

EXPORT_C_FUNC CUresult cuStreamUpdateCaptureDependencies(CUstream hStream, CUgraphNode *dependencies,
                                                         size_t numDependencies, unsigned int flags) {
    INTERCEPT_LOG("intercepted cuStreamUpdateCaptureDependencies");
    using func_ptr = CUresult (*)(CUstream, CUgraphNode *, size_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuStreamUpdateCaptureDependencies"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hStream, dependencies, numDependencies, flags);
}

EXPORT_C_FUNC CUresult cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr, size_t length,
                                              unsigned int flags) {
    INTERCEPT_LOG("intercepted cuStreamAttachMemAsync");
    using func_ptr = CUresult (*)(CUstream, CUdeviceptr, size_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuStreamAttachMemAsync"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hStream, dptr, length, flags);
}

EXPORT_C_FUNC CUresult cuStreamQuery(CUstream hStream) {
    INTERCEPT_LOG("intercepted cuStreamQuery stream(%p)", hStream);
    return xsched::shim::cuda::XStreamQuery(hStream);
}

EXPORT_C_FUNC CUresult cuStreamSynchronize(CUstream hStream) {
    INTERCEPT_LOG("intercepted cuStreamSynchronize stream(%p)", hStream);
    return xsched::shim::cuda::XStreamSynchronize(hStream);
}

EXPORT_C_FUNC CUresult cuStreamDestroy(CUstream hStream) {
    INTERCEPT_LOG("intercepted cuStreamDestroy");
    using func_ptr = CUresult (*)(CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuStreamDestroy"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hStream);
}

// manually add
EXPORT_C_FUNC CUresult cuStreamDestroy_v2(CUstream hStream) {
    INTERCEPT_LOG("intercepted cuStreamDestroy_v2");
    using func_ptr = CUresult (*)(CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuStreamDestroy_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hStream);
}

EXPORT_C_FUNC CUresult cuStreamCopyAttributes(CUstream dst, CUstream src) {
    INTERCEPT_LOG("intercepted cuStreamCopyAttributes");
    using func_ptr = CUresult (*)(CUstream, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuStreamCopyAttributes"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dst, src);
}

EXPORT_C_FUNC CUresult cuStreamGetAttribute(CUstream hStream, CUstreamAttrID attr,
                                            CUstreamAttrValue *value_out) {
    INTERCEPT_LOG("intercepted cuStreamGetAttribute");
    using func_ptr = CUresult (*)(CUstream, CUstreamAttrID, CUstreamAttrValue *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuStreamGetAttribute"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hStream, attr, value_out);
}

EXPORT_C_FUNC CUresult cuStreamSetAttribute(CUstream hStream, CUstreamAttrID attr,
                                            const CUstreamAttrValue *value) {
    INTERCEPT_LOG("intercepted cuStreamSetAttribute");
    using func_ptr = CUresult (*)(CUstream, CUstreamAttrID, const CUstreamAttrValue *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuStreamSetAttribute"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hStream, attr, value);
}

EXPORT_C_FUNC CUresult cuEventCreate(CUevent *phEvent, unsigned int Flags) {
    INTERCEPT_LOG("intercepted cuEventCreate");
    using func_ptr = CUresult (*)(CUevent *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuEventCreate"));
    XASSERT(func_entry, "fail to get func_entry");
    auto ret = func_entry(phEvent, Flags);
    return ret;
}

EXPORT_C_FUNC CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
    INTERCEPT_LOG("intercepted cuEventRecord event(%p), stream(%p)", hEvent, hStream);
    return xsched::shim::cuda::XEventRecord(hEvent, hStream);
}

EXPORT_C_FUNC CUresult cuEventRecordWithFlags(CUevent hEvent, CUstream hStream, unsigned int flags) {
    INTERCEPT_LOG("intercepted cuEventRecordWithFlags event(%p), stream(%p), flags(0x%x)", hEvent, hStream, flags);
    return xsched::shim::cuda::XEventRecordWithFlags(hEvent, hStream, flags);
}

EXPORT_C_FUNC CUresult cuEventQuery(CUevent hEvent) {
    INTERCEPT_LOG("intercepted cuEventQuery event(%p)", hEvent);
    return xsched::shim::cuda::XEventQuery(hEvent);
}

EXPORT_C_FUNC CUresult cuEventSynchronize(CUevent hEvent) {
    INTERCEPT_LOG("intercepted cuEventSynchronize event(%p)", hEvent);
    return xsched::shim::cuda::XEventSynchronize(hEvent);
}

EXPORT_C_FUNC CUresult cuEventDestroy(CUevent hEvent) {
    INTERCEPT_LOG("intercepted cuEventDestroy event(%p)", hEvent);
    return xsched::shim::cuda::XEventDestroy(hEvent);
}

// manually add
EXPORT_C_FUNC CUresult cuEventDestroy_v2(CUevent hEvent) {
    INTERCEPT_LOG("intercepted cuEventDestroy_v2 event(%p)", hEvent);
    return xsched::shim::cuda::XEventDestroy_v2(hEvent);
}

EXPORT_C_FUNC CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd) {
    INTERCEPT_LOG("intercepted cuEventElapsedTime");
    using func_ptr = CUresult (*)(float *, CUevent, CUevent);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuEventElapsedTime"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pMilliseconds, hStart, hEnd);
}

EXPORT_C_FUNC CUresult cuImportExternalMemory(CUexternalMemory *extMem_out,
                                              const CUDA_EXTERNAL_MEMORY_HANDLE_DESC *memHandleDesc) {
    INTERCEPT_LOG("intercepted cuImportExternalMemory");
    using func_ptr = CUresult (*)(CUexternalMemory *, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuImportExternalMemory"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(extMem_out, memHandleDesc);
}

EXPORT_C_FUNC CUresult cuExternalMemoryGetMappedBuffer(
    CUdeviceptr *devPtr, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC *bufferDesc) {
    INTERCEPT_LOG("intercepted cuExternalMemoryGetMappedBuffer");
    using func_ptr = CUresult (*)(CUdeviceptr *, CUexternalMemory, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuExternalMemoryGetMappedBuffer"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(devPtr, extMem, bufferDesc);
}

EXPORT_C_FUNC CUresult cuExternalMemoryGetMappedMipmappedArray(
    CUmipmappedArray *mipmap, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC *mipmapDesc) {
    INTERCEPT_LOG("intercepted cuExternalMemoryGetMappedMipmappedArray");
    using func_ptr =
        CUresult (*)(CUmipmappedArray *, CUexternalMemory, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuExternalMemoryGetMappedMipmappedArray"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(mipmap, extMem, mipmapDesc);
}

EXPORT_C_FUNC CUresult cuDestroyExternalMemory(CUexternalMemory extMem) {
    INTERCEPT_LOG("intercepted cuDestroyExternalMemory");
    using func_ptr = CUresult (*)(CUexternalMemory);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuDestroyExternalMemory"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(extMem);
}

EXPORT_C_FUNC CUresult cuImportExternalSemaphore(
    CUexternalSemaphore *extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC *semHandleDesc) {
    INTERCEPT_LOG("intercepted cuImportExternalSemaphore");
    using func_ptr = CUresult (*)(CUexternalSemaphore *, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuImportExternalSemaphore"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(extSem_out, semHandleDesc);
}

EXPORT_C_FUNC CUresult cuSignalExternalSemaphoresAsync(
    const CUexternalSemaphore *extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *paramsArray,
    unsigned int numExtSems, CUstream stream) {
    INTERCEPT_LOG("intercepted cuSignalExternalSemaphoresAsync");
    using func_ptr = CUresult (*)(const CUexternalSemaphore *, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *,
                                  unsigned int, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuSignalExternalSemaphoresAsync"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(extSemArray, paramsArray, numExtSems, stream);
}

EXPORT_C_FUNC CUresult cuWaitExternalSemaphoresAsync(
    const CUexternalSemaphore *extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *paramsArray,
    unsigned int numExtSems, CUstream stream) {
    INTERCEPT_LOG("intercepted cuWaitExternalSemaphoresAsync");
    using func_ptr =
        CUresult (*)(const CUexternalSemaphore *, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *, unsigned int, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuWaitExternalSemaphoresAsync"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(extSemArray, paramsArray, numExtSems, stream);
}

EXPORT_C_FUNC CUresult cuDestroyExternalSemaphore(CUexternalSemaphore extSem) {
    INTERCEPT_LOG("intercepted cuDestroyExternalSemaphore");
    using func_ptr = CUresult (*)(CUexternalSemaphore);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuDestroyExternalSemaphore"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(extSem);
}

EXPORT_C_FUNC CUresult cuStreamWaitValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value,
                                           unsigned int flags) {
    INTERCEPT_LOG("intercepted cuStreamWaitValue32");
    using func_ptr = CUresult (*)(CUstream, CUdeviceptr, cuuint32_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuStreamWaitValue32"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(stream, addr, value, flags);
}

EXPORT_C_FUNC CUresult cuStreamWaitValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value,
                                           unsigned int flags) {
    INTERCEPT_LOG("intercepted cuStreamWaitValue64");
    using func_ptr = CUresult (*)(CUstream, CUdeviceptr, cuuint64_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuStreamWaitValue64"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(stream, addr, value, flags);
}

EXPORT_C_FUNC CUresult cuStreamWriteValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value,
                                            unsigned int flags) {
    INTERCEPT_LOG("intercepted cuStreamWriteValue32");
    using func_ptr = CUresult (*)(CUstream, CUdeviceptr, cuuint32_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuStreamWriteValue32"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(stream, addr, value, flags);
}

EXPORT_C_FUNC CUresult cuStreamWriteValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value,
                                            unsigned int flags) {
    INTERCEPT_LOG("intercepted cuStreamWriteValue64");
    using func_ptr = CUresult (*)(CUstream, CUdeviceptr, cuuint64_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuStreamWriteValue64"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(stream, addr, value, flags);
}

EXPORT_C_FUNC CUresult cuStreamBatchMemOp(CUstream stream, unsigned int count,
                                          CUstreamBatchMemOpParams *paramArray, unsigned int flags) {
    INTERCEPT_LOG("intercepted cuStreamBatchMemOp");
    using func_ptr = CUresult (*)(CUstream, unsigned int, CUstreamBatchMemOpParams *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuStreamBatchMemOp"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(stream, count, paramArray, flags);
}

EXPORT_C_FUNC CUresult cuFuncGetAttribute(int *pi, CUfunction_attribute attrib, CUfunction hfunc) {
    INTERCEPT_LOG("intercepted cuFuncGetAttribute");
    using func_ptr = CUresult (*)(int *, CUfunction_attribute, CUfunction);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuFuncGetAttribute"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pi, attrib, hfunc);
}

EXPORT_C_FUNC CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value) {
    INTERCEPT_LOG("intercepted cuFuncSetAttribute");
    using func_ptr = CUresult (*)(CUfunction, CUfunction_attribute, int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuFuncSetAttribute"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hfunc, attrib, value);
}

EXPORT_C_FUNC CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) {
    INTERCEPT_LOG("intercepted cuFuncSetCacheConfig");
    using func_ptr = CUresult (*)(CUfunction, CUfunc_cache);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuFuncSetCacheConfig"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hfunc, config);
}

EXPORT_C_FUNC CUresult cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config) {
    INTERCEPT_LOG("intercepted cuFuncSetSharedMemConfig");
    using func_ptr = CUresult (*)(CUfunction, CUsharedconfig);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuFuncSetSharedMemConfig"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hfunc, config);
}

EXPORT_C_FUNC CUresult cuFuncGetModule(CUmodule *hmod, CUfunction hfunc) {
    INTERCEPT_LOG("intercepted cuFuncGetModule");
    using func_ptr = CUresult (*)(CUmodule *, CUfunction);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuFuncGetModule"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hmod, hfunc);
}

EXPORT_C_FUNC CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                                      unsigned int gridDimZ, unsigned int blockDimX,
                                      unsigned int blockDimY, unsigned int blockDimZ,
                                      unsigned int sharedMemBytes, CUstream hStream, void **kernelParams,
                                      void **extra) {
    INTERCEPT_LOG("intercepted cuLaunchKernel func(%p), stream(%p), G[%u, %u, %u] B[%u, %u, %u], params(%p)", f, hStream, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, kernelParams);
    CHECK_CUDA_XQUEUE_REJECT_CALLS

    // if (xsched::shim::cuda::extra::IsNcclFunc(f)) {
    //     XINFO("intercept %s from stream %p", xsched::shim::cuda::extra::GetFuncName(f), hStream);
    // }    

    // if (CudaXQueueQueryReject()) {
    //     if (xsched::shim::cuda::extra::IsNcclFunc(f)) {
    //         XINFO("reject %s from stream %p", xsched::shim::cuda::extra::GetFuncName(f), hStream);
    //     }
    //     return CUDA_SUCCESS;
    // }

    return xsched::shim::cuda::XLaunchKernel(f,
        gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ,
        sharedMemBytes, hStream,
        kernelParams, extra);
}

EXPORT_C_FUNC CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX,
                                                 unsigned int gridDimY, unsigned int gridDimZ,
                                                 unsigned int blockDimX, unsigned int blockDimY,
                                                 unsigned int blockDimZ, unsigned int sharedMemBytes,
                                                 CUstream hStream, void **kernelParams) {
    INTERCEPT_LOG("intercepted cuLaunchCooperativeKernel");
    using func_ptr = CUresult (*)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
                                  unsigned int, unsigned int, CUstream, void **);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuLaunchCooperativeKernel"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream,
                      kernelParams);
}

EXPORT_C_FUNC CUresult cuLaunchCooperativeKernelMultiDevice(CUDA_LAUNCH_PARAMS *launchParamsList,
                                                            unsigned int numDevices, unsigned int flags) {
    INTERCEPT_LOG("intercepted cuLaunchCooperativeKernelMultiDevice");
    using func_ptr = CUresult (*)(CUDA_LAUNCH_PARAMS *, unsigned int, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuLaunchCooperativeKernelMultiDevice"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(launchParamsList, numDevices, flags);
}

EXPORT_C_FUNC CUresult cuLaunchHostFunc(CUstream hStream, CUhostFn fn, void *userData) {
    INTERCEPT_LOG("intercepted cuLaunchHostFunc fn(%p), userData(%p), stream(%p)", fn, userData, hStream);
    return xsched::shim::cuda::XLaunchHostFunc(fn, userData, hStream);
}

EXPORT_C_FUNC CUresult cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z) {
    INTERCEPT_LOG("intercepted cuFuncSetBlockShape");
    using func_ptr = CUresult (*)(CUfunction, int, int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuFuncSetBlockShape"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hfunc, x, y, z);
}

EXPORT_C_FUNC CUresult cuFuncSetSharedSize(CUfunction hfunc, unsigned int bytes) {
    INTERCEPT_LOG("intercepted cuFuncSetSharedSize");
    using func_ptr = CUresult (*)(CUfunction, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuFuncSetSharedSize"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hfunc, bytes);
}

EXPORT_C_FUNC CUresult cuParamSetSize(CUfunction hfunc, unsigned int numbytes) {
    INTERCEPT_LOG("intercepted cuParamSetSize");
    using func_ptr = CUresult (*)(CUfunction, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuParamSetSize"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hfunc, numbytes);
}

EXPORT_C_FUNC CUresult cuParamSeti(CUfunction hfunc, int offset, unsigned int value) {
    INTERCEPT_LOG("intercepted cuParamSeti");
    using func_ptr = CUresult (*)(CUfunction, int, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuParamSeti"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hfunc, offset, value);
}

EXPORT_C_FUNC CUresult cuParamSetf(CUfunction hfunc, int offset, float value) {
    INTERCEPT_LOG("intercepted cuParamSetf");
    using func_ptr = CUresult (*)(CUfunction, int, float);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuParamSetf"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hfunc, offset, value);
}

EXPORT_C_FUNC CUresult cuParamSetv(CUfunction hfunc, int offset, void *ptr, unsigned int numbytes) {
    INTERCEPT_LOG("intercepted cuParamSetv");
    using func_ptr = CUresult (*)(CUfunction, int, void *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuParamSetv"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hfunc, offset, ptr, numbytes);
}

EXPORT_C_FUNC CUresult cuLaunch(CUfunction f) {
    INTERCEPT_LOG("intercepted cuLaunch");
    using func_ptr = CUresult (*)(CUfunction);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuLaunch"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(f);
}

EXPORT_C_FUNC CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height) {
    INTERCEPT_LOG("intercepted cuLaunchGrid");
    using func_ptr = CUresult (*)(CUfunction, int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuLaunchGrid"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(f, grid_width, grid_height);
}

EXPORT_C_FUNC CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height,
                                         CUstream hStream) {
    INTERCEPT_LOG("intercepted cuLaunchGridAsync");
    using func_ptr = CUresult (*)(CUfunction, int, int, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuLaunchGridAsync"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(f, grid_width, grid_height, hStream);
}

EXPORT_C_FUNC CUresult cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef) {
    INTERCEPT_LOG("intercepted cuParamSetTexRef");
    using func_ptr = CUresult (*)(CUfunction, int, CUtexref);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuParamSetTexRef"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hfunc, texunit, hTexRef);
}

EXPORT_C_FUNC CUresult cuGraphCreate(CUgraph *phGraph, unsigned int flags) {
    INTERCEPT_LOG("intercepted cuGraphCreate");
    using func_ptr = CUresult (*)(CUgraph *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphCreate"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(phGraph, flags);
}

EXPORT_C_FUNC CUresult cuGraphAddKernelNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                            const CUgraphNode *dependencies, size_t numDependencies,
                                            const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
    INTERCEPT_LOG("intercepted cuGraphAddKernelNode");
    using func_ptr = CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *, size_t, const CUDA_KERNEL_NODE_PARAMS *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphAddKernelNode"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
}

EXPORT_C_FUNC CUresult cuGraphKernelNodeGetParams(CUgraphNode hNode,
                                                  CUDA_KERNEL_NODE_PARAMS *nodeParams) {
    INTERCEPT_LOG("intercepted cuGraphKernelNodeGetParams");
    using func_ptr = CUresult (*)(CUgraphNode, CUDA_KERNEL_NODE_PARAMS *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphKernelNodeGetParams"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hNode, nodeParams);
}

EXPORT_C_FUNC CUresult cuGraphKernelNodeSetParams(CUgraphNode hNode,
                                                  const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
    INTERCEPT_LOG("intercepted cuGraphKernelNodeSetParams");
    using func_ptr = CUresult (*)(CUgraphNode, const CUDA_KERNEL_NODE_PARAMS *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphKernelNodeSetParams"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hNode, nodeParams);
}

EXPORT_C_FUNC CUresult cuGraphAddMemcpyNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                            const CUgraphNode *dependencies, size_t numDependencies,
                                            const CUDA_MEMCPY3D *copyParams, CUcontext ctx) {
    INTERCEPT_LOG("intercepted cuGraphAddMemcpyNode");
    using func_ptr =
        CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *, size_t, const CUDA_MEMCPY3D *, CUcontext);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphAddMemcpyNode"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(phGraphNode, hGraph, dependencies, numDependencies, copyParams, ctx);
}

EXPORT_C_FUNC CUresult cuGraphMemcpyNodeGetParams(CUgraphNode hNode, CUDA_MEMCPY3D *nodeParams) {
    INTERCEPT_LOG("intercepted cuGraphMemcpyNodeGetParams");
    using func_ptr = CUresult (*)(CUgraphNode, CUDA_MEMCPY3D *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphMemcpyNodeGetParams"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hNode, nodeParams);
}

EXPORT_C_FUNC CUresult cuGraphMemcpyNodeSetParams(CUgraphNode hNode, const CUDA_MEMCPY3D *nodeParams) {
    INTERCEPT_LOG("intercepted cuGraphMemcpyNodeSetParams");
    using func_ptr = CUresult (*)(CUgraphNode, const CUDA_MEMCPY3D *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphMemcpyNodeSetParams"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hNode, nodeParams);
}

EXPORT_C_FUNC CUresult cuGraphAddMemsetNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                            const CUgraphNode *dependencies, size_t numDependencies,
                                            const CUDA_MEMSET_NODE_PARAMS *memsetParams, CUcontext ctx) {
    INTERCEPT_LOG("intercepted cuGraphAddMemsetNode");
    using func_ptr =
        CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *, size_t, const CUDA_MEMSET_NODE_PARAMS *, CUcontext);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphAddMemsetNode"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(phGraphNode, hGraph, dependencies, numDependencies, memsetParams, ctx);
}

EXPORT_C_FUNC CUresult cuGraphMemsetNodeGetParams(CUgraphNode hNode,
                                                  CUDA_MEMSET_NODE_PARAMS *nodeParams) {
    INTERCEPT_LOG("intercepted cuGraphMemsetNodeGetParams");
    using func_ptr = CUresult (*)(CUgraphNode, CUDA_MEMSET_NODE_PARAMS *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphMemsetNodeGetParams"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hNode, nodeParams);
}

EXPORT_C_FUNC CUresult cuGraphMemsetNodeSetParams(CUgraphNode hNode,
                                                  const CUDA_MEMSET_NODE_PARAMS *nodeParams) {
    INTERCEPT_LOG("intercepted cuGraphMemsetNodeSetParams");
    using func_ptr = CUresult (*)(CUgraphNode, const CUDA_MEMSET_NODE_PARAMS *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphMemsetNodeSetParams"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hNode, nodeParams);
}

EXPORT_C_FUNC CUresult cuGraphAddHostNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                          const CUgraphNode *dependencies, size_t numDependencies,
                                          const CUDA_HOST_NODE_PARAMS *nodeParams) {
    INTERCEPT_LOG("intercepted cuGraphAddHostNode");
    using func_ptr = CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *, size_t, const CUDA_HOST_NODE_PARAMS *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphAddHostNode"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
}

EXPORT_C_FUNC CUresult cuGraphHostNodeGetParams(CUgraphNode hNode, CUDA_HOST_NODE_PARAMS *nodeParams) {
    INTERCEPT_LOG("intercepted cuGraphHostNodeGetParams");
    using func_ptr = CUresult (*)(CUgraphNode, CUDA_HOST_NODE_PARAMS *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphHostNodeGetParams"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hNode, nodeParams);
}

EXPORT_C_FUNC CUresult cuGraphHostNodeSetParams(CUgraphNode hNode,
                                                const CUDA_HOST_NODE_PARAMS *nodeParams) {
    INTERCEPT_LOG("intercepted cuGraphHostNodeSetParams");
    using func_ptr = CUresult (*)(CUgraphNode, const CUDA_HOST_NODE_PARAMS *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphHostNodeSetParams"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hNode, nodeParams);
}

EXPORT_C_FUNC CUresult cuGraphAddChildGraphNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                                const CUgraphNode *dependencies, size_t numDependencies,
                                                CUgraph childGraph) {
    INTERCEPT_LOG("intercepted cuGraphAddChildGraphNode");
    using func_ptr = CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *, size_t, CUgraph);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphAddChildGraphNode"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(phGraphNode, hGraph, dependencies, numDependencies, childGraph);
}

EXPORT_C_FUNC CUresult cuGraphChildGraphNodeGetGraph(CUgraphNode hNode, CUgraph *phGraph) {
    INTERCEPT_LOG("intercepted cuGraphChildGraphNodeGetGraph");
    using func_ptr = CUresult (*)(CUgraphNode, CUgraph *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphChildGraphNodeGetGraph"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hNode, phGraph);
}

EXPORT_C_FUNC CUresult cuGraphAddEmptyNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                           const CUgraphNode *dependencies, size_t numDependencies) {
    INTERCEPT_LOG("intercepted cuGraphAddEmptyNode");
    using func_ptr = CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphAddEmptyNode"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(phGraphNode, hGraph, dependencies, numDependencies);
}

EXPORT_C_FUNC CUresult cuGraphAddEventRecordNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                                 const CUgraphNode *dependencies, size_t numDependencies,
                                                 CUevent event) {
    INTERCEPT_LOG("intercepted cuGraphAddEventRecordNode");
    using func_ptr = CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *, size_t, CUevent);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphAddEventRecordNode"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(phGraphNode, hGraph, dependencies, numDependencies, event);
}

EXPORT_C_FUNC CUresult cuGraphEventRecordNodeGetEvent(CUgraphNode hNode, CUevent *event_out) {
    INTERCEPT_LOG("intercepted cuGraphEventRecordNodeGetEvent");
    using func_ptr = CUresult (*)(CUgraphNode, CUevent *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphEventRecordNodeGetEvent"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hNode, event_out);
}

EXPORT_C_FUNC CUresult cuGraphEventRecordNodeSetEvent(CUgraphNode hNode, CUevent event) {
    INTERCEPT_LOG("intercepted cuGraphEventRecordNodeSetEvent");
    using func_ptr = CUresult (*)(CUgraphNode, CUevent);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphEventRecordNodeSetEvent"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hNode, event);
}

EXPORT_C_FUNC CUresult cuGraphAddEventWaitNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                               const CUgraphNode *dependencies, size_t numDependencies,
                                               CUevent event) {
    INTERCEPT_LOG("intercepted cuGraphAddEventWaitNode");
    using func_ptr = CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *, size_t, CUevent);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphAddEventWaitNode"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(phGraphNode, hGraph, dependencies, numDependencies, event);
}

EXPORT_C_FUNC CUresult cuGraphEventWaitNodeGetEvent(CUgraphNode hNode, CUevent *event_out) {
    INTERCEPT_LOG("intercepted cuGraphEventWaitNodeGetEvent");
    using func_ptr = CUresult (*)(CUgraphNode, CUevent *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphEventWaitNodeGetEvent"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hNode, event_out);
}

EXPORT_C_FUNC CUresult cuGraphEventWaitNodeSetEvent(CUgraphNode hNode, CUevent event) {
    INTERCEPT_LOG("intercepted cuGraphEventWaitNodeSetEvent");
    using func_ptr = CUresult (*)(CUgraphNode, CUevent);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphEventWaitNodeSetEvent"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hNode, event);
}

EXPORT_C_FUNC CUresult
cuGraphAddExternalSemaphoresSignalNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies,
                                       size_t numDependencies, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams) {
    INTERCEPT_LOG("intercepted cuGraphAddExternalSemaphoresSignalNode");
    using func_ptr =
        CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *, size_t, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphAddExternalSemaphoresSignalNode"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
}

EXPORT_C_FUNC CUresult
cuGraphExternalSemaphoresSignalNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *params_out) {
    INTERCEPT_LOG("intercepted cuGraphExternalSemaphoresSignalNodeGetParams");
    using func_ptr = CUresult (*)(CUgraphNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphExternalSemaphoresSignalNodeGetParams"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hNode, params_out);
}

EXPORT_C_FUNC CUresult
cuGraphExternalSemaphoresSignalNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams) {
    INTERCEPT_LOG("intercepted cuGraphExternalSemaphoresSignalNodeSetParams");
    using func_ptr = CUresult (*)(CUgraphNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphExternalSemaphoresSignalNodeSetParams"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hNode, nodeParams);
}

EXPORT_C_FUNC CUresult
cuGraphAddExternalSemaphoresWaitNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies,
                                     size_t numDependencies, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams) {
    INTERCEPT_LOG("intercepted cuGraphAddExternalSemaphoresWaitNode");
    using func_ptr =
        CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *, size_t, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphAddExternalSemaphoresWaitNode"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
}

EXPORT_C_FUNC CUresult
cuGraphExternalSemaphoresWaitNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS *params_out) {
    INTERCEPT_LOG("intercepted cuGraphExternalSemaphoresWaitNodeGetParams");
    using func_ptr = CUresult (*)(CUgraphNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphExternalSemaphoresWaitNodeGetParams"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hNode, params_out);
}

EXPORT_C_FUNC CUresult
cuGraphExternalSemaphoresWaitNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams) {
    INTERCEPT_LOG("intercepted cuGraphExternalSemaphoresWaitNodeSetParams");
    using func_ptr = CUresult (*)(CUgraphNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphExternalSemaphoresWaitNodeSetParams"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hNode, nodeParams);
}

EXPORT_C_FUNC CUresult cuGraphAddMemAllocNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                              const CUgraphNode *dependencies, size_t numDependencies,
                                              CUDA_MEM_ALLOC_NODE_PARAMS *nodeParams) {
    INTERCEPT_LOG("intercepted cuGraphAddMemAllocNode");
    using func_ptr = CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *, size_t, CUDA_MEM_ALLOC_NODE_PARAMS *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphAddMemAllocNode"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
}

EXPORT_C_FUNC CUresult cuGraphMemAllocNodeGetParams(CUgraphNode hNode,
                                                    CUDA_MEM_ALLOC_NODE_PARAMS *params_out) {
    INTERCEPT_LOG("intercepted cuGraphMemAllocNodeGetParams");
    using func_ptr = CUresult (*)(CUgraphNode, CUDA_MEM_ALLOC_NODE_PARAMS *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphMemAllocNodeGetParams"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hNode, params_out);
}

EXPORT_C_FUNC CUresult cuGraphAddMemFreeNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                             const CUgraphNode *dependencies, size_t numDependencies,
                                             CUdeviceptr dptr) {
    INTERCEPT_LOG("intercepted cuGraphAddMemFreeNode");
    using func_ptr = CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *, size_t, CUdeviceptr);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphAddMemFreeNode"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(phGraphNode, hGraph, dependencies, numDependencies, dptr);
}

EXPORT_C_FUNC CUresult cuGraphMemFreeNodeGetParams(CUgraphNode hNode, CUdeviceptr *dptr_out) {
    INTERCEPT_LOG("intercepted cuGraphMemFreeNodeGetParams");
    using func_ptr = CUresult (*)(CUgraphNode, CUdeviceptr *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphMemFreeNodeGetParams"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hNode, dptr_out);
}

EXPORT_C_FUNC CUresult cuDeviceGraphMemTrim(CUdevice device) {
    INTERCEPT_LOG("intercepted cuDeviceGraphMemTrim");
    using func_ptr = CUresult (*)(CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuDeviceGraphMemTrim"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(device);
}

EXPORT_C_FUNC CUresult cuDeviceGetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr,
                                                    void *value) {
    INTERCEPT_LOG("intercepted cuDeviceGetGraphMemAttribute");
    using func_ptr = CUresult (*)(CUdevice, CUgraphMem_attribute, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuDeviceGetGraphMemAttribute"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(device, attr, value);
}

EXPORT_C_FUNC CUresult cuDeviceSetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr,
                                                    void *value) {
    INTERCEPT_LOG("intercepted cuDeviceSetGraphMemAttribute");
    using func_ptr = CUresult (*)(CUdevice, CUgraphMem_attribute, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuDeviceSetGraphMemAttribute"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(device, attr, value);
}

EXPORT_C_FUNC CUresult cuGraphClone(CUgraph *phGraphClone, CUgraph originalGraph) {
    INTERCEPT_LOG("intercepted cuGraphClone");
    using func_ptr = CUresult (*)(CUgraph *, CUgraph);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphClone"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(phGraphClone, originalGraph);
}

EXPORT_C_FUNC CUresult cuGraphNodeFindInClone(CUgraphNode *phNode, CUgraphNode hOriginalNode,
                                              CUgraph hClonedGraph) {
    INTERCEPT_LOG("intercepted cuGraphNodeFindInClone");
    using func_ptr = CUresult (*)(CUgraphNode *, CUgraphNode, CUgraph);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphNodeFindInClone"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(phNode, hOriginalNode, hClonedGraph);
}

EXPORT_C_FUNC CUresult cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType *type) {
    INTERCEPT_LOG("intercepted cuGraphNodeGetType");
    using func_ptr = CUresult (*)(CUgraphNode, CUgraphNodeType *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphNodeGetType"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hNode, type);
}

EXPORT_C_FUNC CUresult cuGraphGetNodes(CUgraph hGraph, CUgraphNode *nodes, size_t *numNodes) {
    INTERCEPT_LOG("intercepted cuGraphGetNodes");
    using func_ptr = CUresult (*)(CUgraph, CUgraphNode *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphGetNodes"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hGraph, nodes, numNodes);
}

EXPORT_C_FUNC CUresult cuGraphGetRootNodes(CUgraph hGraph, CUgraphNode *rootNodes, size_t *numRootNodes) {
    INTERCEPT_LOG("intercepted cuGraphGetRootNodes");
    using func_ptr = CUresult (*)(CUgraph, CUgraphNode *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphGetRootNodes"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hGraph, rootNodes, numRootNodes);
}

EXPORT_C_FUNC CUresult cuGraphGetEdges(CUgraph hGraph, CUgraphNode *from, CUgraphNode *to,
                                       size_t *numEdges) {
    INTERCEPT_LOG("intercepted cuGraphGetEdges");
    using func_ptr = CUresult (*)(CUgraph, CUgraphNode *, CUgraphNode *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphGetEdges"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hGraph, from, to, numEdges);
}

EXPORT_C_FUNC CUresult cuGraphNodeGetDependencies(CUgraphNode hNode, CUgraphNode *dependencies,
                                                  size_t *numDependencies) {
    INTERCEPT_LOG("intercepted cuGraphNodeGetDependencies");
    using func_ptr = CUresult (*)(CUgraphNode, CUgraphNode *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphNodeGetDependencies"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hNode, dependencies, numDependencies);
}

EXPORT_C_FUNC CUresult cuGraphNodeGetDependentNodes(CUgraphNode hNode, CUgraphNode *dependentNodes,
                                                    size_t *numDependentNodes) {
    INTERCEPT_LOG("intercepted cuGraphNodeGetDependentNodes");
    using func_ptr = CUresult (*)(CUgraphNode, CUgraphNode *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphNodeGetDependentNodes"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hNode, dependentNodes, numDependentNodes);
}

EXPORT_C_FUNC CUresult cuGraphAddDependencies(CUgraph hGraph, const CUgraphNode *from,
                                              const CUgraphNode *to, size_t numDependencies) {
    INTERCEPT_LOG("intercepted cuGraphAddDependencies");
    using func_ptr = CUresult (*)(CUgraph, const CUgraphNode *, const CUgraphNode *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphAddDependencies"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hGraph, from, to, numDependencies);
}

EXPORT_C_FUNC CUresult cuGraphRemoveDependencies(CUgraph hGraph, const CUgraphNode *from,
                                                 const CUgraphNode *to, size_t numDependencies) {
    INTERCEPT_LOG("intercepted cuGraphRemoveDependencies");
    using func_ptr = CUresult (*)(CUgraph, const CUgraphNode *, const CUgraphNode *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphRemoveDependencies"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hGraph, from, to, numDependencies);
}

EXPORT_C_FUNC CUresult cuGraphDestroyNode(CUgraphNode hNode) {
    INTERCEPT_LOG("intercepted cuGraphDestroyNode");
    using func_ptr = CUresult (*)(CUgraphNode);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphDestroyNode"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hNode);
}

EXPORT_C_FUNC CUresult cuGraphInstantiate(CUgraphExec *phGraphExec, CUgraph hGraph,
                                          CUgraphNode *phErrorNode, char *logBuffer, size_t bufferSize) {
    INTERCEPT_LOG("intercepted cuGraphInstantiate");
    using func_ptr = CUresult (*)(CUgraphExec *, CUgraph, CUgraphNode *, char *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphInstantiate"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(phGraphExec, hGraph, phErrorNode, logBuffer, bufferSize);
}

// manually add
EXPORT_C_FUNC CUresult cuGraphInstantiate_v2(CUgraphExec *phGraphExec, CUgraph hGraph,
                                             CUgraphNode *phErrorNode, char *logBuffer,
                                             size_t bufferSize) {
    INTERCEPT_LOG("intercepted cuGraphInstantiate_v2");
    using func_ptr = CUresult (*)(CUgraphExec *, CUgraph, CUgraphNode *, char *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphInstantiate_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(phGraphExec, hGraph, phErrorNode, logBuffer, bufferSize);
}

EXPORT_C_FUNC CUresult cuGraphInstantiateWithFlags(CUgraphExec *phGraphExec, CUgraph hGraph,
                                                   unsigned long long flags) {
    INTERCEPT_LOG("intercepted cuGraphInstantiateWithFlags");
    using func_ptr = CUresult (*)(CUgraphExec *, CUgraph, unsigned long long);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphInstantiateWithFlags"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(phGraphExec, hGraph, flags);
}

EXPORT_C_FUNC CUresult cuGraphExecGetFlags(CUgraphExec hGraphExec, cuuint64_t *flags) {
    INTERCEPT_LOG("intercepted cuGraphExecGetFlags");
    using func_ptr = CUresult (*)(CUgraphExec, cuuint64_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphExecGetFlags"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hGraphExec, flags);
}

EXPORT_C_FUNC CUresult cuGraphExecKernelNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode,
                                                      const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
    INTERCEPT_LOG("intercepted cuGraphExecKernelNodeSetParams");
    using func_ptr = CUresult (*)(CUgraphExec, CUgraphNode, const CUDA_KERNEL_NODE_PARAMS *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphExecKernelNodeSetParams"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hGraphExec, hNode, nodeParams);
}

EXPORT_C_FUNC CUresult cuGraphExecMemcpyNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode,
                                                      const CUDA_MEMCPY3D *copyParams, CUcontext ctx) {
    INTERCEPT_LOG("intercepted cuGraphExecMemcpyNodeSetParams");
    using func_ptr = CUresult (*)(CUgraphExec, CUgraphNode, const CUDA_MEMCPY3D *, CUcontext);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphExecMemcpyNodeSetParams"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hGraphExec, hNode, copyParams, ctx);
}

EXPORT_C_FUNC CUresult cuGraphExecMemsetNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode,
                                                      const CUDA_MEMSET_NODE_PARAMS *memsetParams,
                                                      CUcontext ctx) {
    INTERCEPT_LOG("intercepted cuGraphExecMemsetNodeSetParams");
    using func_ptr = CUresult (*)(CUgraphExec, CUgraphNode, const CUDA_MEMSET_NODE_PARAMS *, CUcontext);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphExecMemsetNodeSetParams"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hGraphExec, hNode, memsetParams, ctx);
}

EXPORT_C_FUNC CUresult cuGraphExecHostNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode,
                                                    const CUDA_HOST_NODE_PARAMS *nodeParams) {
    INTERCEPT_LOG("intercepted cuGraphExecHostNodeSetParams");
    using func_ptr = CUresult (*)(CUgraphExec, CUgraphNode, const CUDA_HOST_NODE_PARAMS *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphExecHostNodeSetParams"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hGraphExec, hNode, nodeParams);
}

EXPORT_C_FUNC CUresult cuGraphExecChildGraphNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode,
                                                          CUgraph childGraph) {
    INTERCEPT_LOG("intercepted cuGraphExecChildGraphNodeSetParams");
    using func_ptr = CUresult (*)(CUgraphExec, CUgraphNode, CUgraph);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphExecChildGraphNodeSetParams"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hGraphExec, hNode, childGraph);
}

EXPORT_C_FUNC CUresult cuGraphExecEventRecordNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode,
                                                          CUevent event) {
    INTERCEPT_LOG("intercepted cuGraphExecEventRecordNodeSetEvent");
    using func_ptr = CUresult (*)(CUgraphExec, CUgraphNode, CUevent);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphExecEventRecordNodeSetEvent"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hGraphExec, hNode, event);
}

EXPORT_C_FUNC CUresult cuGraphExecEventWaitNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode,
                                                        CUevent event) {
    INTERCEPT_LOG("intercepted cuGraphExecEventWaitNodeSetEvent");
    using func_ptr = CUresult (*)(CUgraphExec, CUgraphNode, CUevent);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphExecEventWaitNodeSetEvent"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hGraphExec, hNode, event);
}

EXPORT_C_FUNC CUresult cuGraphExecExternalSemaphoresSignalNodeSetParams(
    CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams) {
    INTERCEPT_LOG("intercepted cuGraphExecExternalSemaphoresSignalNodeSetParams");
    using func_ptr = CUresult (*)(CUgraphExec, CUgraphNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphExecExternalSemaphoresSignalNodeSetParams"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hGraphExec, hNode, nodeParams);
}

EXPORT_C_FUNC CUresult cuGraphExecExternalSemaphoresWaitNodeSetParams(
    CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams) {
    INTERCEPT_LOG("intercepted cuGraphExecExternalSemaphoresWaitNodeSetParams");
    using func_ptr = CUresult (*)(CUgraphExec, CUgraphNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphExecExternalSemaphoresWaitNodeSetParams"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hGraphExec, hNode, nodeParams);
}

EXPORT_C_FUNC CUresult cuGraphUpload(CUgraphExec hGraphExec, CUstream hStream) {
    INTERCEPT_LOG("intercepted cuGraphUpload");
    using func_ptr = CUresult (*)(CUgraphExec, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphUpload"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hGraphExec, hStream);
}

EXPORT_C_FUNC CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) {
    INTERCEPT_LOG("intercepted cuGraphLaunch");
    using func_ptr = CUresult (*)(CUgraphExec, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphLaunch"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hGraphExec, hStream);
}

EXPORT_C_FUNC CUresult cuGraphExecDestroy(CUgraphExec hGraphExec) {
    INTERCEPT_LOG("intercepted cuGraphExecDestroy");
    using func_ptr = CUresult (*)(CUgraphExec);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphExecDestroy"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hGraphExec);
}

EXPORT_C_FUNC CUresult cuGraphDestroy(CUgraph hGraph) {
    INTERCEPT_LOG("intercepted cuGraphDestroy");
    using func_ptr = CUresult (*)(CUgraph);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphDestroy"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hGraph);
}

EXPORT_C_FUNC CUresult cuGraphExecUpdate(CUgraphExec hGraphExec, CUgraph hGraph,
                                         CUgraphNode *hErrorNode_out,
                                         CUgraphExecUpdateResult *updateResult_out) {
    INTERCEPT_LOG("intercepted cuGraphExecUpdate");
    using func_ptr = CUresult (*)(CUgraphExec, CUgraph, CUgraphNode *, CUgraphExecUpdateResult *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphExecUpdate"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hGraphExec, hGraph, hErrorNode_out, updateResult_out);
}

// EXPORT_C_FUNC CUresult cuGraphExecUpdate_v2(CUgraphExec hGraphExec, CUgraph hGraph,
//                                             CUgraphExecUpdateResultInfo *resultInfo) {
//     INTERCEPT_LOG("intercepted cuGraphExecUpdate_v2");
//     using func_ptr = CUresult (*)(CUgraphExec, CUgraph, CUgraphExecUpdateResultInfo *);
//     static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphExecUpdate_v2"));
//     XASSERT(func_entry, "fail to get func_entry");
//     return func_entry(hGraphExec, hGraph, resultInfo);
// }

EXPORT_C_FUNC CUresult cuStreamGetId(CUstream hStream, unsigned long long *streamId) {
    INTERCEPT_LOG("intercepted cuStreamGetId");
    using func_ptr = CUresult (*)(CUstream, unsigned long long *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuStreamGetId"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hStream, streamId);
}

EXPORT_C_FUNC CUresult cuGraphKernelNodeCopyAttributes(CUgraphNode dst, CUgraphNode src) {
    INTERCEPT_LOG("intercepted cuGraphKernelNodeCopyAttributes");
    using func_ptr = CUresult (*)(CUgraphNode, CUgraphNode);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphKernelNodeCopyAttributes"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dst, src);
}

EXPORT_C_FUNC CUresult cuGraphKernelNodeGetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr,
                                                     CUkernelNodeAttrValue *value_out) {
    INTERCEPT_LOG("intercepted cuGraphKernelNodeGetAttribute");
    using func_ptr = CUresult (*)(CUgraphNode, CUkernelNodeAttrID, CUkernelNodeAttrValue *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphKernelNodeGetAttribute"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hNode, attr, value_out);
}

EXPORT_C_FUNC CUresult cuGraphKernelNodeSetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr,
                                                     const CUkernelNodeAttrValue *value) {
    INTERCEPT_LOG("intercepted cuGraphKernelNodeSetAttribute");
    using func_ptr = CUresult (*)(CUgraphNode, CUkernelNodeAttrID, const CUkernelNodeAttrValue *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphKernelNodeSetAttribute"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hNode, attr, value);
}

EXPORT_C_FUNC CUresult cuGraphDebugDotPrint(CUgraph hGraph, const char *path, unsigned int flags) {
    INTERCEPT_LOG("intercepted cuGraphDebugDotPrint");
    using func_ptr = CUresult (*)(CUgraph, const char *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphDebugDotPrint"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hGraph, path, flags);
}

EXPORT_C_FUNC CUresult cuUserObjectCreate(CUuserObject *object_out, void *ptr, CUhostFn destroy,
                                          unsigned int initialRefcount, unsigned int flags) {
    INTERCEPT_LOG("intercepted cuUserObjectCreate");
    using func_ptr = CUresult (*)(CUuserObject *, void *, CUhostFn, unsigned int, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuUserObjectCreate"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(object_out, ptr, destroy, initialRefcount, flags);
}

EXPORT_C_FUNC CUresult cuUserObjectRetain(CUuserObject object, unsigned int count) {
    INTERCEPT_LOG("intercepted cuUserObjectRetain");
    using func_ptr = CUresult (*)(CUuserObject, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuUserObjectRetain"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(object, count);
}

EXPORT_C_FUNC CUresult cuUserObjectRelease(CUuserObject object, unsigned int count) {
    INTERCEPT_LOG("intercepted cuUserObjectRelease");
    using func_ptr = CUresult (*)(CUuserObject, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuUserObjectRelease"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(object, count);
}

EXPORT_C_FUNC CUresult cuGraphRetainUserObject(CUgraph graph, CUuserObject object, unsigned int count,
                                               unsigned int flags) {
    INTERCEPT_LOG("intercepted cuGraphRetainUserObject");
    using func_ptr = CUresult (*)(CUgraph, CUuserObject, unsigned int, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphRetainUserObject"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(graph, object, count, flags);
}

EXPORT_C_FUNC CUresult cuGraphReleaseUserObject(CUgraph graph, CUuserObject object, unsigned int count) {
    INTERCEPT_LOG("intercepted cuGraphReleaseUserObject");
    using func_ptr = CUresult (*)(CUgraph, CUuserObject, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphReleaseUserObject"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(graph, object, count);
}

EXPORT_C_FUNC CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, CUfunction func,
                                                                   int blockSize,
                                                                   size_t dynamicSMemSize) {
    INTERCEPT_LOG("intercepted cuOccupancyMaxActiveBlocksPerMultiprocessor");
    using func_ptr = CUresult (*)(int *, CUfunction, int, size_t);
    static auto func_entry =
        reinterpret_cast<func_ptr>(GetCudaSymbol("cuOccupancyMaxActiveBlocksPerMultiprocessor"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(numBlocks, func, blockSize, dynamicSMemSize);
}

EXPORT_C_FUNC CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags) {
    INTERCEPT_LOG("intercepted cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
    using func_ptr = CUresult (*)(int *, CUfunction, int, size_t, unsigned int);
    static auto func_entry =
        reinterpret_cast<func_ptr>(GetCudaSymbol("cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(numBlocks, func, blockSize, dynamicSMemSize, flags);
}

EXPORT_C_FUNC CUresult cuOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize, CUfunction func,
                                                        CUoccupancyB2DSize blockSizeToDynamicSMemSize,
                                                        size_t dynamicSMemSize, int blockSizeLimit) {
    INTERCEPT_LOG("intercepted cuOccupancyMaxPotentialBlockSize");
    using func_ptr = CUresult (*)(int *, int *, CUfunction, CUoccupancyB2DSize, size_t, int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuOccupancyMaxPotentialBlockSize"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit);
}

EXPORT_C_FUNC CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(
    int *minGridSize, int *blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize,
    size_t dynamicSMemSize, int blockSizeLimit, unsigned int flags) {
    INTERCEPT_LOG("intercepted cuOccupancyMaxPotentialBlockSizeWithFlags");
    using func_ptr = CUresult (*)(int *, int *, CUfunction, CUoccupancyB2DSize, size_t, int, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuOccupancyMaxPotentialBlockSizeWithFlags"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit, flags);
}

EXPORT_C_FUNC CUresult cuOccupancyAvailableDynamicSMemPerBlock(size_t *dynamicSmemSize, CUfunction func,
                                                               int numBlocks, int blockSize) {
    INTERCEPT_LOG("intercepted cuOccupancyAvailableDynamicSMemPerBlock");
    using func_ptr = CUresult (*)(size_t *, CUfunction, int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuOccupancyAvailableDynamicSMemPerBlock"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dynamicSmemSize, func, numBlocks, blockSize);
}

EXPORT_C_FUNC CUresult cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, unsigned int Flags) {
    INTERCEPT_LOG("intercepted cuTexRefSetArray");
    using func_ptr = CUresult (*)(CUtexref, CUarray, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuTexRefSetArray"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hTexRef, hArray, Flags);
}

EXPORT_C_FUNC CUresult cuTexRefSetMipmappedArray(CUtexref hTexRef, CUmipmappedArray hMipmappedArray,
                                                 unsigned int Flags) {
    INTERCEPT_LOG("intercepted cuTexRefSetMipmappedArray");
    using func_ptr = CUresult (*)(CUtexref, CUmipmappedArray, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuTexRefSetMipmappedArray"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hTexRef, hMipmappedArray, Flags);
}

EXPORT_C_FUNC CUresult cuTexRefSetAddress(size_t *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr,
                                          size_t bytes) {
    INTERCEPT_LOG("intercepted cuTexRefSetAddress");
    using func_ptr = CUresult (*)(size_t *, CUtexref, CUdeviceptr, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuTexRefSetAddress"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(ByteOffset, hTexRef, dptr, bytes);
}

// manually add
EXPORT_C_FUNC CUresult cuTexRefSetAddress_v2(size_t *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr,
                                             size_t bytes) {
    INTERCEPT_LOG("intercepted cuTexRefSetAddress_v2");
    using func_ptr = CUresult (*)(size_t *, CUtexref, CUdeviceptr, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuTexRefSetAddress_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(ByteOffset, hTexRef, dptr, bytes);
}

EXPORT_C_FUNC CUresult cuTexRefSetAddress2D(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc,
                                            CUdeviceptr dptr, size_t Pitch) {
    INTERCEPT_LOG("intercepted cuTexRefSetAddress2D");
    using func_ptr = CUresult (*)(CUtexref, const CUDA_ARRAY_DESCRIPTOR *, CUdeviceptr, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuTexRefSetAddress2D"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hTexRef, desc, dptr, Pitch);
}

// manually add
EXPORT_C_FUNC CUresult cuTexRefSetAddress2D_v3(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc,
                                               CUdeviceptr dptr, size_t Pitch) {
    INTERCEPT_LOG("intercepted cuTexRefSetAddress2D_v3");
    using func_ptr = CUresult (*)(CUtexref, const CUDA_ARRAY_DESCRIPTOR *, CUdeviceptr, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuTexRefSetAddress2D_v3"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hTexRef, desc, dptr, Pitch);
}

EXPORT_C_FUNC CUresult cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents) {
    INTERCEPT_LOG("intercepted cuTexRefSetFormat");
    using func_ptr = CUresult (*)(CUtexref, CUarray_format, int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuTexRefSetFormat"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hTexRef, fmt, NumPackedComponents);
}

EXPORT_C_FUNC CUresult cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am) {
    INTERCEPT_LOG("intercepted cuTexRefSetAddressMode");
    using func_ptr = CUresult (*)(CUtexref, int, CUaddress_mode);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuTexRefSetAddressMode"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hTexRef, dim, am);
}

EXPORT_C_FUNC CUresult cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm) {
    INTERCEPT_LOG("intercepted cuTexRefSetFilterMode");
    using func_ptr = CUresult (*)(CUtexref, CUfilter_mode);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuTexRefSetFilterMode"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hTexRef, fm);
}

EXPORT_C_FUNC CUresult cuTexRefSetMipmapFilterMode(CUtexref hTexRef, CUfilter_mode fm) {
    INTERCEPT_LOG("intercepted cuTexRefSetMipmapFilterMode");
    using func_ptr = CUresult (*)(CUtexref, CUfilter_mode);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuTexRefSetMipmapFilterMode"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hTexRef, fm);
}

EXPORT_C_FUNC CUresult cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias) {
    INTERCEPT_LOG("intercepted cuTexRefSetMipmapLevelBias");
    using func_ptr = CUresult (*)(CUtexref, float);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuTexRefSetMipmapLevelBias"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hTexRef, bias);
}

EXPORT_C_FUNC CUresult cuTexRefSetMipmapLevelClamp(CUtexref hTexRef, float minMipmapLevelClamp,
                                                   float maxMipmapLevelClamp) {
    INTERCEPT_LOG("intercepted cuTexRefSetMipmapLevelClamp");
    using func_ptr = CUresult (*)(CUtexref, float, float);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuTexRefSetMipmapLevelClamp"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hTexRef, minMipmapLevelClamp, maxMipmapLevelClamp);
}

EXPORT_C_FUNC CUresult cuTexRefSetMaxAnisotropy(CUtexref hTexRef, unsigned int maxAniso) {
    INTERCEPT_LOG("intercepted cuTexRefSetMaxAnisotropy");
    using func_ptr = CUresult (*)(CUtexref, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuTexRefSetMaxAnisotropy"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hTexRef, maxAniso);
}

EXPORT_C_FUNC CUresult cuTexRefSetBorderColor(CUtexref hTexRef, float *pBorderColor) {
    INTERCEPT_LOG("intercepted cuTexRefSetBorderColor");
    using func_ptr = CUresult (*)(CUtexref, float *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuTexRefSetBorderColor"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hTexRef, pBorderColor);
}

EXPORT_C_FUNC CUresult cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags) {
    INTERCEPT_LOG("intercepted cuTexRefSetFlags");
    using func_ptr = CUresult (*)(CUtexref, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuTexRefSetFlags"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hTexRef, Flags);
}

EXPORT_C_FUNC CUresult cuTexRefGetAddress(CUdeviceptr *pdptr, CUtexref hTexRef) {
    INTERCEPT_LOG("intercepted cuTexRefGetAddress");
    using func_ptr = CUresult (*)(CUdeviceptr *, CUtexref);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuTexRefGetAddress"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pdptr, hTexRef);
}

// manually add
EXPORT_C_FUNC CUresult cuTexRefGetAddress_v2(CUdeviceptr *pdptr, CUtexref hTexRef) {
    INTERCEPT_LOG("intercepted cuTexRefGetAddress_v2");
    using func_ptr = CUresult (*)(CUdeviceptr *, CUtexref);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuTexRefGetAddress_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pdptr, hTexRef);
}

EXPORT_C_FUNC CUresult cuTexRefGetArray(CUarray *phArray, CUtexref hTexRef) {
    INTERCEPT_LOG("intercepted cuTexRefGetArray");
    using func_ptr = CUresult (*)(CUarray *, CUtexref);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuTexRefGetArray"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(phArray, hTexRef);
}

EXPORT_C_FUNC CUresult cuTexRefGetMipmappedArray(CUmipmappedArray *phMipmappedArray, CUtexref hTexRef) {
    INTERCEPT_LOG("intercepted cuTexRefGetMipmappedArray");
    using func_ptr = CUresult (*)(CUmipmappedArray *, CUtexref);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuTexRefGetMipmappedArray"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(phMipmappedArray, hTexRef);
}

EXPORT_C_FUNC CUresult cuTexRefGetAddressMode(CUaddress_mode *pam, CUtexref hTexRef, int dim) {
    INTERCEPT_LOG("intercepted cuTexRefGetAddressMode");
    using func_ptr = CUresult (*)(CUaddress_mode *, CUtexref, int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuTexRefGetAddressMode"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pam, hTexRef, dim);
}

EXPORT_C_FUNC CUresult cuTexRefGetFilterMode(CUfilter_mode *pfm, CUtexref hTexRef) {
    INTERCEPT_LOG("intercepted cuTexRefGetFilterMode");
    using func_ptr = CUresult (*)(CUfilter_mode *, CUtexref);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuTexRefGetFilterMode"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pfm, hTexRef);
}

EXPORT_C_FUNC CUresult cuTexRefGetFormat(CUarray_format *pFormat, int *pNumChannels, CUtexref hTexRef) {
    INTERCEPT_LOG("intercepted cuTexRefGetFormat");
    using func_ptr = CUresult (*)(CUarray_format *, int *, CUtexref);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuTexRefGetFormat"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pFormat, pNumChannels, hTexRef);
}

EXPORT_C_FUNC CUresult cuTexRefGetMipmapFilterMode(CUfilter_mode *pfm, CUtexref hTexRef) {
    INTERCEPT_LOG("intercepted cuTexRefGetMipmapFilterMode");
    using func_ptr = CUresult (*)(CUfilter_mode *, CUtexref);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuTexRefGetMipmapFilterMode"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pfm, hTexRef);
}

EXPORT_C_FUNC CUresult cuTexRefGetMipmapLevelBias(float *pbias, CUtexref hTexRef) {
    INTERCEPT_LOG("intercepted cuTexRefGetMipmapLevelBias");
    using func_ptr = CUresult (*)(float *, CUtexref);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuTexRefGetMipmapLevelBias"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pbias, hTexRef);
}

EXPORT_C_FUNC CUresult cuTexRefGetMipmapLevelClamp(float *pminMipmapLevelClamp,
                                                   float *pmaxMipmapLevelClamp, CUtexref hTexRef) {
    INTERCEPT_LOG("intercepted cuTexRefGetMipmapLevelClamp");
    using func_ptr = CUresult (*)(float *, float *, CUtexref);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuTexRefGetMipmapLevelClamp"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pminMipmapLevelClamp, pmaxMipmapLevelClamp, hTexRef);
}

EXPORT_C_FUNC CUresult cuTexRefGetMaxAnisotropy(int *pmaxAniso, CUtexref hTexRef) {
    INTERCEPT_LOG("intercepted cuTexRefGetMaxAnisotropy");
    using func_ptr = CUresult (*)(int *, CUtexref);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuTexRefGetMaxAnisotropy"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pmaxAniso, hTexRef);
}

EXPORT_C_FUNC CUresult cuTexRefGetBorderColor(float *pBorderColor, CUtexref hTexRef) {
    INTERCEPT_LOG("intercepted cuTexRefGetBorderColor");
    using func_ptr = CUresult (*)(float *, CUtexref);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuTexRefGetBorderColor"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pBorderColor, hTexRef);
}

EXPORT_C_FUNC CUresult cuTexRefGetFlags(unsigned int *pFlags, CUtexref hTexRef) {
    INTERCEPT_LOG("intercepted cuTexRefGetFlags");
    using func_ptr = CUresult (*)(unsigned int *, CUtexref);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuTexRefGetFlags"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pFlags, hTexRef);
}

EXPORT_C_FUNC CUresult cuTexRefCreate(CUtexref *pTexRef) {
    INTERCEPT_LOG("intercepted cuTexRefCreate");
    using func_ptr = CUresult (*)(CUtexref *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuTexRefCreate"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pTexRef);
}

EXPORT_C_FUNC CUresult cuTexRefDestroy(CUtexref hTexRef) {
    INTERCEPT_LOG("intercepted cuTexRefDestroy");
    using func_ptr = CUresult (*)(CUtexref);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuTexRefDestroy"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hTexRef);
}

EXPORT_C_FUNC CUresult cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray, unsigned int Flags) {
    INTERCEPT_LOG("intercepted cuSurfRefSetArray");
    using func_ptr = CUresult (*)(CUsurfref, CUarray, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuSurfRefSetArray"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hSurfRef, hArray, Flags);
}

EXPORT_C_FUNC CUresult cuSurfRefGetArray(CUarray *phArray, CUsurfref hSurfRef) {
    INTERCEPT_LOG("intercepted cuSurfRefGetArray");
    using func_ptr = CUresult (*)(CUarray *, CUsurfref);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuSurfRefGetArray"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(phArray, hSurfRef);
}

EXPORT_C_FUNC CUresult cuTexObjectCreate(CUtexObject *pTexObject, const CUDA_RESOURCE_DESC *pResDesc,
                                         const CUDA_TEXTURE_DESC *pTexDesc,
                                         const CUDA_RESOURCE_VIEW_DESC *pResViewDesc) {
    INTERCEPT_LOG("intercepted cuTexObjectCreate");
    using func_ptr = CUresult (*)(CUtexObject *, const CUDA_RESOURCE_DESC *, const CUDA_TEXTURE_DESC *,
                                  const CUDA_RESOURCE_VIEW_DESC *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuTexObjectCreate"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pTexObject, pResDesc, pTexDesc, pResViewDesc);
}

EXPORT_C_FUNC CUresult cuTexObjectDestroy(CUtexObject texObject) {
    INTERCEPT_LOG("intercepted cuTexObjectDestroy");
    using func_ptr = CUresult (*)(CUtexObject);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuTexObjectDestroy"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(texObject);
}

EXPORT_C_FUNC CUresult cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc, CUtexObject texObject) {
    INTERCEPT_LOG("intercepted cuTexObjectGetResourceDesc");
    using func_ptr = CUresult (*)(CUDA_RESOURCE_DESC *, CUtexObject);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuTexObjectGetResourceDesc"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pResDesc, texObject);
}

EXPORT_C_FUNC CUresult cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC *pTexDesc, CUtexObject texObject) {
    INTERCEPT_LOG("intercepted cuTexObjectGetTextureDesc");
    using func_ptr = CUresult (*)(CUDA_TEXTURE_DESC *, CUtexObject);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuTexObjectGetTextureDesc"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pTexDesc, texObject);
}

EXPORT_C_FUNC CUresult cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC *pResViewDesc,
                                                      CUtexObject texObject) {
    INTERCEPT_LOG("intercepted cuTexObjectGetResourceViewDesc");
    using func_ptr = CUresult (*)(CUDA_RESOURCE_VIEW_DESC *, CUtexObject);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuTexObjectGetResourceViewDesc"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pResViewDesc, texObject);
}

EXPORT_C_FUNC CUresult cuSurfObjectCreate(CUsurfObject *pSurfObject, const CUDA_RESOURCE_DESC *pResDesc) {
    INTERCEPT_LOG("intercepted cuSurfObjectCreate");
    using func_ptr = CUresult (*)(CUsurfObject *, const CUDA_RESOURCE_DESC *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuSurfObjectCreate"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pSurfObject, pResDesc);
}

EXPORT_C_FUNC CUresult cuSurfObjectDestroy(CUsurfObject surfObject) {
    INTERCEPT_LOG("intercepted cuSurfObjectDestroy");
    using func_ptr = CUresult (*)(CUsurfObject);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuSurfObjectDestroy"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(surfObject);
}

EXPORT_C_FUNC CUresult cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc,
                                                   CUsurfObject surfObject) {
    INTERCEPT_LOG("intercepted cuSurfObjectGetResourceDesc");
    using func_ptr = CUresult (*)(CUDA_RESOURCE_DESC *, CUsurfObject);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuSurfObjectGetResourceDesc"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pResDesc, surfObject);
}

EXPORT_C_FUNC CUresult cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev, CUdevice peerDev) {
    INTERCEPT_LOG("intercepted cuDeviceCanAccessPeer");
    using func_ptr = CUresult (*)(int *, CUdevice, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuDeviceCanAccessPeer"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(canAccessPeer, dev, peerDev);
}

EXPORT_C_FUNC CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags) {
    INTERCEPT_LOG("intercepted cuCtxEnablePeerAccess");
    using func_ptr = CUresult (*)(CUcontext, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuCtxEnablePeerAccess"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(peerContext, Flags);
}

EXPORT_C_FUNC CUresult cuCtxDisablePeerAccess(CUcontext peerContext) {
    INTERCEPT_LOG("intercepted cuCtxDisablePeerAccess");
    using func_ptr = CUresult (*)(CUcontext);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuCtxDisablePeerAccess"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(peerContext);
}

EXPORT_C_FUNC CUresult cuDeviceGetP2PAttribute(int *value, CUdevice_P2PAttribute attrib,
                                               CUdevice srcDevice, CUdevice dstDevice) {
    INTERCEPT_LOG("intercepted cuDeviceGetP2PAttribute");
    using func_ptr = CUresult (*)(int *, CUdevice_P2PAttribute, CUdevice, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuDeviceGetP2PAttribute"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(value, attrib, srcDevice, dstDevice);
}

EXPORT_C_FUNC CUresult cuGraphicsUnregisterResource(CUgraphicsResource resource) {
    INTERCEPT_LOG("intercepted cuGraphicsUnregisterResource");
    using func_ptr = CUresult (*)(CUgraphicsResource);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphicsUnregisterResource"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(resource);
}

EXPORT_C_FUNC CUresult cuGraphicsSubResourceGetMappedArray(CUarray *pArray, CUgraphicsResource resource,
                                                           unsigned int arrayIndex,
                                                           unsigned int mipLevel) {
    INTERCEPT_LOG("intercepted cuGraphicsSubResourceGetMappedArray");
    using func_ptr = CUresult (*)(CUarray *, CUgraphicsResource, unsigned int, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphicsSubResourceGetMappedArray"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pArray, resource, arrayIndex, mipLevel);
}

EXPORT_C_FUNC CUresult cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray *pMipmappedArray,
                                                                 CUgraphicsResource resource) {
    INTERCEPT_LOG("intercepted cuGraphicsResourceGetMappedMipmappedArray");
    using func_ptr = CUresult (*)(CUmipmappedArray *, CUgraphicsResource);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphicsResourceGetMappedMipmappedArray"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pMipmappedArray, resource);
}

EXPORT_C_FUNC CUresult cuGraphicsResourceGetMappedPointer(CUdeviceptr *pDevPtr, size_t *pSize,
                                                          CUgraphicsResource resource) {
    INTERCEPT_LOG("intercepted cuGraphicsResourceGetMappedPointer");
    using func_ptr = CUresult (*)(CUdeviceptr *, size_t *, CUgraphicsResource);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphicsResourceGetMappedPointer"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pDevPtr, pSize, resource);
}

// manually add
EXPORT_C_FUNC CUresult cuGraphicsResourceGetMappedPointer_v2(CUdeviceptr *pDevPtr, size_t *pSize,
                                                             CUgraphicsResource resource) {
    INTERCEPT_LOG("intercepted cuGraphicsResourceGetMappedPointer_v2");
    using func_ptr = CUresult (*)(CUdeviceptr *, size_t *, CUgraphicsResource);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphicsResourceGetMappedPointer_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pDevPtr, pSize, resource);
}

EXPORT_C_FUNC CUresult cuGraphicsResourceSetMapFlags(CUgraphicsResource resource, unsigned int flags) {
    INTERCEPT_LOG("intercepted cuGraphicsResourceSetMapFlags");
    using func_ptr = CUresult (*)(CUgraphicsResource, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphicsResourceSetMapFlags"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(resource, flags);
}

// manually add
EXPORT_C_FUNC CUresult cuGraphicsResourceSetMapFlags_v2(CUgraphicsResource resource, unsigned int flags) {
    INTERCEPT_LOG("intercepted cuGraphicsResourceSetMapFlags_v2");
    using func_ptr = CUresult (*)(CUgraphicsResource, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphicsResourceSetMapFlags_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(resource, flags);
}

EXPORT_C_FUNC CUresult cuGraphicsMapResources(unsigned int count, CUgraphicsResource *resources,
                                              CUstream hStream) {
    INTERCEPT_LOG("intercepted cuGraphicsMapResources");
    using func_ptr = CUresult (*)(unsigned int, CUgraphicsResource *, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphicsMapResources"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(count, resources, hStream);
}

EXPORT_C_FUNC CUresult cuGraphicsUnmapResources(unsigned int count, CUgraphicsResource *resources,
                                                CUstream hStream) {
    INTERCEPT_LOG("intercepted cuGraphicsUnmapResources");
    using func_ptr = CUresult (*)(unsigned int, CUgraphicsResource *, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphicsUnmapResources"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(count, resources, hStream);
}

// manually delete
// EXPORT_C_FUNC CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion,
//                                         cuuint64_t flags) {
//     INTERCEPT_LOG("intercepted cuGetProcAddress");
//     using func_ptr = CUresult (*)(const char *, void **, int, cuuint64_t);
//     static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGetProcAddress"));
//     XASSERT(func_entry, "fail to get func_entry");
//     return func_entry(symbol, pfn, cudaVersion, flags);
// }

EXPORT_C_FUNC CUresult cuGetExportTable(const void **ppExportTable, const CUuuid *pExportTableId) {
    INTERCEPT_LOG("intercepted cuGetExportTable");
    using func_ptr = CUresult (*)(const void **, const CUuuid *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGetExportTable"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(ppExportTable, pExportTableId);
}

EXPORT_C_FUNC CUresult cuTexRefSetAddress2D_v2(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc,
                                               CUdeviceptr dptr, size_t Pitch) {
    INTERCEPT_LOG("intercepted cuTexRefSetAddress2D_v2");
    using func_ptr = CUresult (*)(CUtexref, const CUDA_ARRAY_DESCRIPTOR *, CUdeviceptr, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuTexRefSetAddress2D_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hTexRef, desc, dptr, Pitch);
}

EXPORT_C_FUNC CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
    INTERCEPT_LOG("intercepted cuMemcpyHtoD_v2");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(CUdeviceptr, const void *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemcpyHtoD_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dstDevice, srcHost, ByteCount);
}

EXPORT_C_FUNC CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    INTERCEPT_LOG("intercepted cuMemcpyDtoH_v2");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(void *, CUdeviceptr, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemcpyDtoH_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dstHost, srcDevice, ByteCount);
}

EXPORT_C_FUNC CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
    INTERCEPT_LOG("intercepted cuMemcpyDtoD_v2");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(CUdeviceptr, CUdeviceptr, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemcpyDtoD_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dstDevice, srcDevice, ByteCount);
}

EXPORT_C_FUNC CUresult cuMemcpyDtoA_v2(CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice,
                                       size_t ByteCount) {
    INTERCEPT_LOG("intercepted cuMemcpyDtoA_v2");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(CUarray, size_t, CUdeviceptr, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemcpyDtoA_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dstArray, dstOffset, srcDevice, ByteCount);
}

EXPORT_C_FUNC CUresult cuMemcpyAtoD_v2(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset,
                                       size_t ByteCount) {
    INTERCEPT_LOG("intercepted cuMemcpyAtoD_v2");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(CUdeviceptr, CUarray, size_t, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemcpyAtoD_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dstDevice, srcArray, srcOffset, ByteCount);
}

EXPORT_C_FUNC CUresult cuMemcpyHtoA_v2(CUarray dstArray, size_t dstOffset, const void *srcHost,
                                       size_t ByteCount) {
    INTERCEPT_LOG("intercepted cuMemcpyHtoA_v2");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(CUarray, size_t, const void *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemcpyHtoA_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dstArray, dstOffset, srcHost, ByteCount);
}

EXPORT_C_FUNC CUresult cuMemcpyAtoH_v2(void *dstHost, CUarray srcArray, size_t srcOffset,
                                       size_t ByteCount) {
    INTERCEPT_LOG("intercepted cuMemcpyAtoH_v2");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(void *, CUarray, size_t, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemcpyAtoH_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dstHost, srcArray, srcOffset, ByteCount);
}

EXPORT_C_FUNC CUresult cuMemcpyAtoA_v2(CUarray dstArray, size_t dstOffset, CUarray srcArray,
                                       size_t srcOffset, size_t ByteCount) {
    INTERCEPT_LOG("intercepted cuMemcpyAtoA_v2");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(CUarray, size_t, CUarray, size_t, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemcpyAtoA_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dstArray, dstOffset, srcArray, srcOffset, ByteCount);
}

EXPORT_C_FUNC CUresult cuMemcpyHtoAAsync_v2(CUarray dstArray, size_t dstOffset, const void *srcHost,
                                            size_t ByteCount, CUstream hStream) {
    INTERCEPT_LOG("intercepted cuMemcpyHtoAAsync_v2");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(CUarray, size_t, const void *, size_t, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemcpyHtoAAsync_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dstArray, dstOffset, srcHost, ByteCount, hStream);
}

EXPORT_C_FUNC CUresult cuMemcpyAtoHAsync_v2(void *dstHost, CUarray srcArray, size_t srcOffset,
                                            size_t ByteCount, CUstream hStream) {
    INTERCEPT_LOG("intercepted cuMemcpyAtoHAsync_v2");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(void *, CUarray, size_t, size_t, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemcpyAtoHAsync_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dstHost, srcArray, srcOffset, ByteCount, hStream);
}

EXPORT_C_FUNC CUresult cuMemcpy2D_v2(const CUDA_MEMCPY2D *pCopy) {
    INTERCEPT_LOG("intercepted cuMemcpy2D_v2");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(const CUDA_MEMCPY2D *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemcpy2D_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pCopy);
}

EXPORT_C_FUNC CUresult cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D *pCopy) {
    INTERCEPT_LOG("intercepted cuMemcpy2DUnaligned_v2");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(const CUDA_MEMCPY2D *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemcpy2DUnaligned_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pCopy);
}

EXPORT_C_FUNC CUresult cuMemcpy3D_v2(const CUDA_MEMCPY3D *pCopy) {
    INTERCEPT_LOG("intercepted cuMemcpy3D_v2");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(const CUDA_MEMCPY3D *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemcpy3D_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(pCopy);
}

EXPORT_C_FUNC CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount,
                                            CUstream hStream) {
    INTERCEPT_LOG("intercepted cuMemcpyHtoDAsync_v2 dst(0x%llx) src(%p) size(%lu)", dstDevice, srcHost, ByteCount);
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    return xsched::shim::cuda::XMemcpyHtoDAsync_v2(dstDevice, srcHost, ByteCount, hStream);
}

EXPORT_C_FUNC CUresult cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount,
                                            CUstream hStream) {
    INTERCEPT_LOG("intercepted cuMemcpyDtoHAsync_v2 dst(%p) src(0x%llx) size(%lu)", dstHost, srcDevice, ByteCount);
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    return xsched::shim::cuda::XMemcpyDtoHAsync_v2(dstHost, srcDevice, ByteCount, hStream);
}

EXPORT_C_FUNC CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                                            size_t ByteCount, CUstream hStream) {
    INTERCEPT_LOG("intercepted cuMemcpyDtoDAsync_v2 dst(0x%llx) src(0x%llx) size(%lu)", dstDevice, srcDevice, ByteCount);
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    return xsched::shim::cuda::XMemcpyDtoDAsync_v2(dstDevice, srcDevice, ByteCount, hStream);
}

EXPORT_C_FUNC CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D *pCopy, CUstream hStream) {
    INTERCEPT_LOG("intercepted cuMemcpy2DAsync_v2 copy(%p) stream(%p)", pCopy, hStream);
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    return xsched::shim::cuda::XMemcpy2DAsync_v2(pCopy, hStream);
}

EXPORT_C_FUNC CUresult cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D *pCopy, CUstream hStream) {
    INTERCEPT_LOG("intercepted cuMemcpy3DAsync_v2 copy(%p) stream(%p)", pCopy, hStream);
    return xsched::shim::cuda::XMemcpy3DAsync_v2(pCopy, hStream);
}

EXPORT_C_FUNC CUresult cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
    INTERCEPT_LOG("intercepted cuMemsetD8_v2");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(CUdeviceptr, unsigned char, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemsetD8_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dstDevice, uc, N);
}

EXPORT_C_FUNC CUresult cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us, size_t N) {
    INTERCEPT_LOG("intercepted cuMemsetD16_v2");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(CUdeviceptr, unsigned short, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemsetD16_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dstDevice, us, N);
}

EXPORT_C_FUNC CUresult cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
    INTERCEPT_LOG("intercepted cuMemsetD32_v2");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(CUdeviceptr, unsigned int, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemsetD32_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dstDevice, ui, N);
}

EXPORT_C_FUNC CUresult cuMemsetD2D8_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc,
                                       size_t Width, size_t Height) {
    INTERCEPT_LOG("intercepted cuMemsetD2D8_v2");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(CUdeviceptr, size_t, unsigned char, size_t, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemsetD2D8_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dstDevice, dstPitch, uc, Width, Height);
}

EXPORT_C_FUNC CUresult cuMemsetD2D16_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us,
                                        size_t Width, size_t Height) {
    INTERCEPT_LOG("intercepted cuMemsetD2D16_v2");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(CUdeviceptr, size_t, unsigned short, size_t, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemsetD2D16_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dstDevice, dstPitch, us, Width, Height);
}

EXPORT_C_FUNC CUresult cuMemsetD2D32_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui,
                                        size_t Width, size_t Height) {
    INTERCEPT_LOG("intercepted cuMemsetD2D32_v2");
    CHECK_CUDA_XQUEUE_REJECT_CALLS
    using func_ptr = CUresult (*)(CUdeviceptr, size_t, unsigned int, size_t, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemsetD2D32_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(dstDevice, dstPitch, ui, Width, Height);
}

EXPORT_C_FUNC CUresult cuStreamBeginCapture_ptsz(CUstream hStream) {
    INTERCEPT_LOG("intercepted cuStreamBeginCapture_ptsz");
    using func_ptr = CUresult (*)(CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuStreamBeginCapture_ptsz"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hStream);
}

EXPORT_C_FUNC CUresult cuStreamBeginCapture_v2(CUstream hStream, CUstreamCaptureMode mode) {
    INTERCEPT_LOG("intercepted cuStreamBeginCapture_v2");
    using func_ptr = CUresult (*)(CUstream, CUstreamCaptureMode);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuStreamBeginCapture_v2"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(hStream, mode);
}

EXPORT_C_FUNC CUresult cuGetProcAddress_ptsz(const char *symbol, void **funcPtr, int driverVersion,
                                             cuuint64_t flags) {
    INTERCEPT_LOG("intercepted cuGetProcAddress_ptsz");
    using func_ptr = CUresult (*)(const char *, void **, int, cuuint64_t);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGetProcAddress_ptsz"));
    XASSERT(func_entry, "fail to get func_entry");
    return func_entry(symbol, funcPtr, driverVersion, flags);
}

/////////////// added to support CUDA12.3 ///////////////
EXPORT_C_FUNC CUresult cuKernelGetFunction(CUfunction *pFunc, CUkernel kernel) {
    INTERCEPT_LOG("intercepted cuKernelGetFunction");
    using func_ptr = CUresult (*)(CUfunction *, CUkernel);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuKernelGetFunction"));
    XASSERT(func_entry != nullptr, "Failed to get function address");
    return func_entry(pFunc, kernel);
}

EXPORT_C_FUNC CUresult cuKernelGetAttribute(int *pi, CUfunction_attribute attrib, CUkernel kernel, CUdevice dev) {
    INTERCEPT_LOG("intercepted cuKernelGetAttribute");
    using func_ptr = CUresult (*)(int *, CUfunction_attribute, CUkernel, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuKernelGetAttribute"));
    XASSERT(func_entry != nullptr, "Failed to get function address");
    return func_entry(pi, attrib, kernel, dev);
}

EXPORT_C_FUNC CUresult cuKernelSetAttribute(CUfunction_attribute attrib, int val, CUkernel kernel, CUdevice dev) {
    INTERCEPT_LOG("intercepted cuKernelSetAttribute");
    using func_ptr = CUresult (*)(CUfunction_attribute, int, CUkernel, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuKernelSetAttribute"));
    XASSERT(func_entry != nullptr, "Failed to get function address");
    return func_entry(attrib, val, kernel, dev);
}

EXPORT_C_FUNC CUresult cuKernelSetCacheConfig(CUkernel kernel, CUfunc_cache config, CUdevice dev) {
    INTERCEPT_LOG("intercepted cuKernelSetCacheConfig");
    using func_ptr = CUresult (*)(CUkernel, CUfunc_cache, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuKernelSetCacheConfig"));
    XASSERT(func_entry != nullptr, "Failed to get function address");
    return func_entry(kernel, config, dev);
}

EXPORT_C_FUNC CUresult cuKernelGetName(const char **name, CUkernel hfunc) {
    INTERCEPT_LOG("intercepted cuKernelGetName");
    using func_ptr = CUresult (*)(const char **, CUkernel);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuKernelGetName"));
    XASSERT(func_entry != nullptr, "Failed to get function address");
    return func_entry(name, hfunc);
}

EXPORT_C_FUNC CUresult cuFuncGetName(const char **name, CUfunction hfunc) {
    INTERCEPT_LOG("intercepted cuFuncGetName");
    using func_ptr = CUresult (*)(const char**, CUfunction);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuFuncGetName"));
    XASSERT(func_entry != nullptr, "Failed to get function address");
    return func_entry(name, hfunc);
}

EXPORT_C_FUNC CUresult cuArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS *memoryRequirements, CUarray array, CUdevice device) {
    INTERCEPT_LOG("intercepted cuArrayGetMemoryRequirements");
    using func_ptr = CUresult (*)(CUDA_ARRAY_MEMORY_REQUIREMENTS *, CUarray, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuArrayGetMemoryRequirements"));
    XASSERT(func_entry != nullptr, "Failed to get function address");
    return func_entry(memoryRequirements, array, device);
}

EXPORT_C_FUNC CUresult cuMipmappedArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS *memoryRequirements, CUmipmappedArray mipmap, CUdevice device) {
    INTERCEPT_LOG("intercepted cuMipmappedArrayGetMemoryRequirements");
    using func_ptr = CUresult (*)(CUDA_ARRAY_MEMORY_REQUIREMENTS *, CUmipmappedArray, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMipmappedArrayGetMemoryRequirements"));
    XASSERT(func_entry != nullptr, "Failed to get function address");
    return func_entry(memoryRequirements, mipmap, device);
}

EXPORT_C_FUNC CUresult cuLaunchKernelEx(const CUlaunchConfig *config, CUfunction f, void **kernelParams, void **extra) {
    INTERCEPT_LOG("intercepted cuLaunchKernelEx");
    using func_ptr = CUresult (*)(const CUlaunchConfig *, CUfunction, void **, void **);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuLaunchKernelEx"));
    XASSERT(func_entry != nullptr, "Failed to get function address");
    return func_entry(config, f, kernelParams, extra);
}

EXPORT_C_FUNC CUresult cuOccupancyMaxPotentialClusterSize(int *clusterSize, CUfunction func, const CUlaunchConfig *config) {
    INTERCEPT_LOG("intercepted cuOccupancyMaxPotentialClusterSize");
    using func_ptr = CUresult (*)(int *, CUfunction, const CUlaunchConfig *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuOccupancyMaxPotentialClusterSize"));
    XASSERT(func_entry != nullptr, "Failed to get function address");
    return func_entry(clusterSize, func, config);
}

EXPORT_C_FUNC CUresult cuOccupancyMaxActiveClusters(int *numClusters, CUfunction func, const CUlaunchConfig *config) {
    INTERCEPT_LOG("intercepted cuOccupancyMaxActiveClusters");
    using func_ptr = CUresult (*)(int *, CUfunction, const CUlaunchConfig *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuOccupancyMaxActiveClusters"));
    XASSERT(func_entry != nullptr, "Failed to get function address");
    return func_entry(numClusters, func, config);
}

EXPORT_C_FUNC CUresult cuMemAdvise_v2(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUmemLocation location) {
    INTERCEPT_LOG("intercepted cuMemAdvise_v2");
    using func_ptr = CUresult (*)(CUdeviceptr, size_t, CUmem_advise, CUmemLocation);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemAdvise_v2"));
    XASSERT(func_entry != nullptr, "Failed to get function address");
    return func_entry(devPtr, count, advice, location);
}

EXPORT_C_FUNC CUresult cuMemPrefetchAsync_v2(CUdeviceptr devPtr, size_t count, CUmemLocation location, unsigned int flags, CUstream hStream) {
    INTERCEPT_LOG("intercepted cuMemPrefetchAsync_v2");
    using func_ptr = CUresult (*)(CUdeviceptr, size_t, CUmemLocation, unsigned int, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuMemPrefetchAsync_v2"));
    XASSERT(func_entry != nullptr, "Failed to get function address");
    return func_entry(devPtr, count, location, flags, hStream);
}

EXPORT_C_FUNC CUresult cuGraphGetEdges_v2(CUgraph hGraph, CUgraphNode *from, CUgraphNode *to, CUgraphEdgeData *edgeData, size_t *numEdges) {
    INTERCEPT_LOG("intercepted cuGraphGetEdges_v2");
    using func_ptr = CUresult (*)(CUgraph, CUgraphNode *, CUgraphNode *, CUgraphEdgeData *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphGetEdges_v2"));
    XASSERT(func_entry != nullptr, "Failed to get function address");
    return func_entry(hGraph, from, to, edgeData, numEdges);
}

EXPORT_C_FUNC CUresult cuGraphNodeGetDependencies_v2(CUgraphNode hNode, CUgraphNode *dependencies, CUgraphEdgeData *edgeData, size_t *numDependencies) {
    INTERCEPT_LOG("intercepted cuGraphNodeGetDependencies_v2");
    using func_ptr = CUresult (*)(CUgraphNode, CUgraphNode *, CUgraphEdgeData *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(GetCudaSymbol("cuGraphNodeGetDependencies_v2"));
    XASSERT(func_entry != nullptr, "Failed to get function address");
    return func_entry(hNode, dependencies, edgeData, numDependencies);
}
