#include "utils/xassert.h"
#include "cuxtra/cuxtra.h"
#include "hal/cuda/driver.h"
#include "shim/xdag/shim.h"
#include "shim/xdag/xctrl.h"
#include "shim/xdag/agent.h"
#include "shim/xdag/handle.h"
#include "shim/xdag/manager.h"
#include "sched/policy/hpf.h"

static cudlaDevHandle g_last_created_dla_device = nullptr;

namespace xsched::shim::xdag
{

CUresult XCtxCreateV2(CUcontext *pctx, unsigned int flags, CUdevice dev)
{
    XASSERT(pctx != nullptr, "pctx is nullptr");
    *pctx = cuXtraCreateContextJetson(dev, flags);
    return CUDA_SUCCESS;
}

cudlaStatus XCreateDevice(uint64_t device, cudlaDevHandle *dev_handle, uint32_t flags)
{
    XASSERT(dev_handle != nullptr, "dev_handle is nullptr");
    cudlaDevHandle handle = cuXtraCreateDevHandleDla(device, flags);
    *dev_handle = handle;
    g_last_created_dla_device = handle;
    XDEBG("created DLA device handle %p on core %lu", handle, device);
    return cudlaSuccess;
}

} // namespace xsched::shim::xdag

using namespace xsched::hal;
using namespace xsched::sched;
using namespace xsched::preempt;
using namespace xsched::shim::xdag;

static std::mutex g_device_map_mtx;
static std::map<XQueueHandle, XDevice> g_device_map;

XDAG_CTRL_FUNC void XDagEnableCudaContext(CUcontext context)
{
    if (context == nullptr) {
        XASSERT(cuda::Driver::CtxGetCurrent(&context) == CUDA_SUCCESS,
                "Failed to get current context");
    }

    CudaContextManager::EnableManager(context);
    XQueueHandle handle = GetXHandleForCuda(context);

    g_device_map_mtx.lock();
    g_device_map[handle] = XDevice::kDeviceCUDA;
    g_device_map_mtx.unlock();

    auto e = std::make_unique<XQueueCreateEvent>(handle, XDevice::kDeviceCUDA);
    g_xdag_agent.RecvEvent(std::move(e));
}

XDAG_CTRL_FUNC void XDagSetPriorityCuda(Prio priority, CUcontext context)
{
    if (context == nullptr) {
        XASSERT(cuda::Driver::CtxGetCurrent(&context) == CUDA_SUCCESS,
                "Failed to get current context");
    }
    XQueueHandle handle = GetXHandleForCuda(context);
    auto hint = std::make_unique<xsched::sched::SetPriorityHint>(
        (xsched::sched::Prio)priority, handle);
    g_xdag_agent.GiveHint(std::move(hint));
}

XDAG_CTRL_FUNC void XDagTaskBeginCuda(CUcontext context)
{
    if (context == nullptr) {
        XASSERT(cuda::Driver::CtxGetCurrent(&context) == CUDA_SUCCESS,
                "Failed to get current context");
    }

    CudaContextManager::GetManager(context)->TaskBegin();

    XQueueHandle handle = GetXHandleForCuda(context);
    auto e = std::make_unique<XQueueReadyEvent>(handle);
    g_xdag_agent.RecvEvent(std::move(e));
}

XDAG_CTRL_FUNC void XDagTaskEndCuda(CUcontext context)
{
    if (context == nullptr) {
        XASSERT(cuda::Driver::CtxGetCurrent(&context) == CUDA_SUCCESS,
                "Failed to get current context");
    }

    CudaContextManager::GetManager(context)->TaskEnd();

    XQueueHandle handle = GetXHandleForCuda(context);
    auto e = std::make_unique<XQueueIdleEvent>(handle);
    g_xdag_agent.RecvEvent(std::move(e));
}

XDAG_CTRL_FUNC void XDagSuspendCuda(CUcontext context)
{
    CudaContextManager::GetManager(context)->Suspend();
}

XDAG_CTRL_FUNC void XDagResumeCuda(CUcontext context)
{
    CudaContextManager::GetManager(context)->Resume();
}

XDAG_CTRL_FUNC cudlaDevHandle XDagGetLastCreatedDlaDevice()
{
    return g_last_created_dla_device;
}

XDAG_CTRL_FUNC void XDagEnableDlaDevice(cudlaDevHandle dev, uint64_t dev_no)
{
    DlaDeviceManager::EnableManager(dev);
    XQueueHandle handle = GetXHandleForDla(dev);

    g_device_map_mtx.lock();
    g_device_map[handle] = XDevice::kDeviceCUDLA;
    g_device_map_mtx.unlock();

    auto e = std::make_unique<XQueueCreateEvent>(
        handle, XDevice(XDevice::kDeviceCUDLA + dev_no));
    g_xdag_agent.RecvEvent(std::move(e));
}

XDAG_CTRL_FUNC void XDagSetPriorityDla(Prio priority, cudlaDevHandle dev)
{
    XQueueHandle handle = GetXHandleForDla(dev);
    auto hint = std::make_unique<xsched::sched::SetPriorityHint>(
        (xsched::sched::Prio)priority, handle);
    g_xdag_agent.GiveHint(std::move(hint));
}

XDAG_CTRL_FUNC void XDagTaskBeginDla(cudlaDevHandle dev)
{
    DlaDeviceManager::GetManager(dev)->TaskBegin();

    XQueueHandle handle = GetXHandleForDla(dev);
    auto e = std::make_unique<XQueueReadyEvent>(handle);
    g_xdag_agent.RecvEvent(std::move(e));
}

XDAG_CTRL_FUNC void XDagTaskEndDla(cudlaDevHandle dev)
{
    DlaDeviceManager::GetManager(dev)->TaskEnd();

    XQueueHandle handle = GetXHandleForDla(dev);
    auto e = std::make_unique<XQueueIdleEvent>(handle);
    g_xdag_agent.RecvEvent(std::move(e));
}

XDAG_CTRL_FUNC void XDagSuspendDla(cudlaDevHandle dev)
{
    DlaDeviceManager::GetManager(dev)->Suspend();
}

XDAG_CTRL_FUNC void XDagResumeDla(cudlaDevHandle dev)
{
    DlaDeviceManager::GetManager(dev)->Resume();
}

XDAG_CTRL_FUNC void XDagSuspend(uint64_t xhandle)
{
    std::unique_lock<std::mutex> lock(g_device_map_mtx);
    auto it = g_device_map.find(xhandle);
    if (it == g_device_map.end()) return;
    XDevice device = it->second;
    lock.unlock();

    if (device == XDevice::kDeviceCUDA) {
        XDagSuspendCuda(GetCudaContext(xhandle));
    } else if (device == XDevice::kDeviceCUDLA) {
        XDagSuspendDla(GetDlaDevice(xhandle));
    }
}

XDAG_CTRL_FUNC void XDagResume(uint64_t xhandle)
{
    std::unique_lock<std::mutex> lock(g_device_map_mtx);
    auto it = g_device_map.find(xhandle);
    if (it == g_device_map.end()) return;
    XDevice device = it->second;
    lock.unlock();

    if (device == XDevice::kDeviceCUDA) {
        XDagResumeCuda(GetCudaContext(xhandle));
    } else if (device == XDevice::kDeviceCUDLA) {
        XDagResumeDla(GetDlaDevice(xhandle));
    }
}
