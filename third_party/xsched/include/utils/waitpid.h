#pragma once

#include <list>
#include <vector>
#include <memory>
#include <mutex>
#include <thread>
#include <functional>
#include <unordered_map>

#include "utils/common.h"

namespace xsched::utils
{

enum EpollEventType
{
    kEpollEventTerminate = 0,
    kEpollEventPid       = 1,
};

typedef std::function<void (PID)> TerminateCallback;

class PidWaiter
{
public:
    PidWaiter(TerminateCallback callback);
    ~PidWaiter();

    void Start();
    void Stop();
    void AddWait(PID pid);

private:
    void WaitWorker();
    PID GetEventPid(uint64_t data);
    EpollEventType GetEventType(uint64_t data);
    uint64_t PackEventData(EpollEventType type, PID pid);

    int event_fd_ = -1;
    int epoll_fd_ = -1;

    std::mutex mtx_;
    std::unordered_map<PID, int> pid_fds_;

    TerminateCallback callback_;
    std::unique_ptr<std::thread> thread_ = nullptr;
};

} // namespace xsched::utils
