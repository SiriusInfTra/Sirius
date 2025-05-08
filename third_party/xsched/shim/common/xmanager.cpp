#include <list>
#include <mutex>
#include <memory>
#include <unordered_map>

#include "utils/xassert.h"
#include "shim/common/agent.h"
#include "shim/common/xmanager.h"

using namespace xsched::shim;
using namespace xsched::preempt;

static std::mutex xqueue_mutex;
static std::unordered_map<XQueueHandle, std::shared_ptr<XQueue>> xqueues;

std::shared_ptr<XQueue>
XManager::CreateXQueue(std::shared_ptr<HalQueue> hal_queue,
                       XQueueHandle handle,
                       XDevice device,
                       XPreemptMode mode,
                       int64_t queue_length,
                       int64_t sync_interval)
{
    xqueue_mutex.lock();
    auto it = xqueues.find(handle);
    XASSERT(it == xqueues.end(),
            "XQueue with handle 0x%lx already created", handle);

    auto xqueue = std::make_shared<XQueue>(
        hal_queue, handle, device, mode, queue_length, sync_interval);
    xqueues[handle] = xqueue;
    xqueue_mutex.unlock();

    return xqueue;
}

void XManager::DestroyXQueue(XQueueHandle handle)
{
    xqueue_mutex.lock();
    auto it = xqueues.find(handle);
    XASSERT(it != xqueues.end(),
            "XQueue with handle 0x%lx does not exist", handle);
    auto xqueue = it->second;
    xqueues.erase(it);
    xqueue_mutex.unlock();

    xqueue = nullptr;
}

std::shared_ptr<XQueue> XManager::GetXQueue(XQueueHandle handle)
{
    std::unique_lock<std::mutex> lock(xqueue_mutex);
    auto it = xqueues.find(handle);

    if (it == xqueues.end()) return nullptr;
    return it->second;
}

bool XManager::Submit(std::shared_ptr<HalCommand> hal_command,
                      XQueueHandle handle)
{
    std::shared_ptr<XQueue> xqueue;
    
    xqueue_mutex.lock();
    auto it = xqueues.find(handle);
    if (it == xqueues.end()) goto fallback;
    xqueue = it->second;
    xqueue_mutex.unlock();

    xqueue->Submit(hal_command);
    return true;

fallback:
    xqueue_mutex.unlock();
    return false;
}

bool XManager::Synchronize(XQueueHandle handle)
{
    std::shared_ptr<XQueue> xqueue;
    
    xqueue_mutex.lock();
    auto it = xqueues.find(handle);
    if (it == xqueues.end()) goto fallback;
    xqueue = it->second;
    xqueue_mutex.unlock();

    xqueue->Synchronize();
    return true;

fallback:
    xqueue_mutex.unlock();
    return false;
}

void XManager::SynchronizeAllXQueues()
{
    std::list<std::shared_ptr<XCommand>> sync_commands;

    xqueue_mutex.lock();
    for (auto it : xqueues) {
        sync_commands.emplace_back(it.second->EnqueueSynchronizeCommand());
    }
    xqueue_mutex.unlock();

    for (auto sync_command : sync_commands) {
        sync_command->Synchronize();
    }
}

void XManager::Suspend(XQueueHandle handle, bool sync_hal_queue)
{
    xqueue_mutex.lock();
    auto it = xqueues.find(handle);
    XASSERT(it != xqueues.end(),
            "XQueue with handle 0x%lx does not exist", handle);
    auto xqueue = it->second;
    xqueue_mutex.unlock();

    xqueue->Suspend(sync_hal_queue);
}

void XManager::Resume(XQueueHandle handle, bool drop_commands)
{
    xqueue_mutex.lock();
    auto it = xqueues.find(handle);
    XASSERT(it != xqueues.end(),
            "XQueue with handle 0x%lx does not exist", handle);
    auto xqueue = it->second;
    xqueue_mutex.unlock();

    xqueue->Resume(drop_commands);
}

XQueueState XManager::GetXQueueState(XQueueHandle handle)
{
    std::unique_lock<std::mutex> lock(xqueue_mutex);
    auto it = xqueues.find(handle);
    if (it == xqueues.end()) return kQueueStateUnknown;
    
    auto xqueue = it->second;
    lock.unlock();

    return xqueue->GetState();
}

void XManager::GiveHint(std::unique_ptr<const sched::Hint> hint)
{
    g_sched_agent.GiveHint(std::move(hint));
}
