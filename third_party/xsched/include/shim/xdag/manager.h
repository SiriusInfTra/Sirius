#pragma once

#include <map>
#include <mutex>

#include "hal/cuda/driver.h"
#include "hal/cudla/driver.h"

namespace xsched::shim::xdag
{

class CudaContextManager
{
public:
    CudaContextManager(CUcontext ctx);
    ~CudaContextManager() = default;

    void TaskBegin();
    void TaskEnd();

    void Suspend();
    void Resume();

    static void EnableManager(CUcontext ctx);
    static CudaContextManager *GetManager(CUcontext ctx);

private:
    void Update();

    std::mutex mtx_;
    bool ready_ = false;
    bool suspended_ = false;
    bool should_suspend_ = false;
    CUcontext context_ = nullptr;

    static std::mutex mgr_map_mtx_;
    static std::map <CUcontext, CudaContextManager *> mgr_map_;
};

class DlaDeviceManager
{
public:
    DlaDeviceManager(cudlaDevHandle dev);
    ~DlaDeviceManager() = default;

    void TaskBegin();
    void TaskEnd();

    void Suspend();
    void Resume();

    static void EnableManager(cudlaDevHandle dev);
    static DlaDeviceManager *GetManager(cudlaDevHandle dev);

private:
    void Update();

    std::mutex mtx_;
    bool ready_ = false;
    bool suspended_ = false;
    bool should_suspend_ = false;
    cudlaDevHandle device_ = nullptr;

    static std::mutex mgr_map_mtx_;
    static std::map <cudlaDevHandle, DlaDeviceManager *> mgr_map_;
};

} // namespace xsched::shim::xdag
