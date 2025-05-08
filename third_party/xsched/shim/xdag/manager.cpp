#include "utils/log.h"
#include "utils/common.h"
#include "cuxtra/cuxtra.h"
#include "shim/xdag/manager.h"

namespace xsched::shim::xdag
{

std::mutex CudaContextManager::mgr_map_mtx_;
std::map <CUcontext, CudaContextManager *> CudaContextManager::mgr_map_;

CudaContextManager::CudaContextManager(CUcontext ctx)
    : context_(ctx)
{
    cuXtraRegisterContextJetson(context_);
}

void CudaContextManager::EnableManager(CUcontext ctx)
{
    std::unique_lock<std::mutex> lock(mgr_map_mtx_);
    auto it = mgr_map_.find(ctx);
    if (it != mgr_map_.end()) return;

    CudaContextManager *manager = new CudaContextManager(ctx);
    mgr_map_[ctx] = manager;
}

CudaContextManager *CudaContextManager::GetManager(CUcontext ctx)
{
    std::unique_lock<std::mutex> lock(mgr_map_mtx_);
    auto it = mgr_map_.find(ctx);
    XASSERT(it != mgr_map_.end(),
            "CudaContextManager of CUcontext %p not found, is it registered?",
            ctx);
    return it->second;
}

void CudaContextManager::TaskBegin()
{
    std::unique_lock<std::mutex> lock(mtx_);
    ready_ = true;
    Update();
}

void CudaContextManager::TaskEnd()
{
    std::unique_lock<std::mutex> lock(mtx_);
    ready_ = false;
    Update();
}
    
void CudaContextManager::Suspend()
{
    std::unique_lock<std::mutex> lock(mtx_);
    should_suspend_ = true;
    Update();
}

void CudaContextManager::Resume()
{
    std::unique_lock<std::mutex> lock(mtx_);
    should_suspend_ = false;
    Update();
}

void CudaContextManager::Update()
{
    if (ready_) {
        // If the context is in ready state (has a task to run),
        // the suspension state should be the same as should_suspend_.
        if (should_suspend_ == suspended_) return;
        if (should_suspend_) {
            suspended_ = true;
            cuXtraSuspendContextJetson(context_);
        } else {
            suspended_ = false;
            cuXtraResumeContextJetson(context_);
        }
    } else {
        // If the context is in block state (no task to run),
        // the suspension state should be false.
        if (suspended_) {
            suspended_ = false;
            cuXtraResumeContextJetson(context_);
        }
    }
}

std::mutex DlaDeviceManager::mgr_map_mtx_;
std::map <cudlaDevHandle, DlaDeviceManager *> DlaDeviceManager::mgr_map_;

DlaDeviceManager::DlaDeviceManager(cudlaDevHandle dev)
    : device_(dev)
{
    
}

void DlaDeviceManager::EnableManager(cudlaDevHandle dev)
{
    std::unique_lock<std::mutex> lock(mgr_map_mtx_);
    auto it = mgr_map_.find(dev);
    if (it != mgr_map_.end()) return;

    DlaDeviceManager *manager = new DlaDeviceManager(dev);
    mgr_map_[dev] = manager;
}

DlaDeviceManager *DlaDeviceManager::GetManager(cudlaDevHandle dev)
{
    std::unique_lock<std::mutex> lock(mgr_map_mtx_);
    auto it = mgr_map_.find(dev);
    XASSERT(it != mgr_map_.end(),
            "DlaDeviceManager of cudlaDevHandle %p not found, is it registered?",
            dev);
    return it->second;
}

void DlaDeviceManager::TaskBegin()
{
    std::unique_lock<std::mutex> lock(mtx_);
    ready_ = true;
    Update();
}

void DlaDeviceManager::TaskEnd()
{
    std::unique_lock<std::mutex> lock(mtx_);
    ready_ = false;
    Update();
}
    
void DlaDeviceManager::Suspend()
{
    std::unique_lock<std::mutex> lock(mtx_);
    should_suspend_ = true;
    Update();
}

void DlaDeviceManager::Resume()
{
    std::unique_lock<std::mutex> lock(mtx_);
    should_suspend_ = false;
    Update();
}

void DlaDeviceManager::Update()
{
    if (ready_) {
        // If the device is in ready state (has a task to run),
        // the suspension state should be the same as should_suspend_.
        if (should_suspend_ == suspended_) return;
        if (should_suspend_) {
            suspended_ = true;
            cuXtraSuspendDla(device_);
        } else {
            suspended_ = false;
            cuXtraResumeDla(device_);
        }
    } else {
        // If the device is in block state (no task to run),
        // the suspension state should be false.
        if (suspended_) {
            suspended_ = false;
            cuXtraResumeDla(device_);
        }
    }
}

} // namespace xsched::shim::xdag
