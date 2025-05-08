#include <sys/epoll.h>
#include <sys/eventfd.h>

#include "utils/log.h"
#include "utils/common.h"
#include "utils/xassert.h"
#include "utils/waitpid.h"

using namespace xsched::utils;

PidWaiter::PidWaiter(TerminateCallback callback)
    : callback_(callback)
{

}

PidWaiter::~PidWaiter()
{
    Stop();
}

void PidWaiter::Start()
{
    event_fd_ = eventfd(0, EFD_CLOEXEC);
    XASSERT(event_fd_ >= 0, "fail to create event fd");
    epoll_fd_ = epoll_create1(EPOLL_CLOEXEC);
    XASSERT(epoll_fd_ >= 0, "fail to create epoll fd");

    struct epoll_event ev;
    ev.events = EPOLLIN;
    ev.data.u64 = PackEventData(kEpollEventTerminate, 0);
    XASSERT(!epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, event_fd_, &ev),
            "fail to add event fd to epoll");

    thread_ = std::make_unique<std::thread>(&PidWaiter::WaitWorker, this);
}

void PidWaiter::Stop()
{
    if (thread_ != nullptr) {
        XASSERT(!eventfd_write(event_fd_, 1), "fail to write event fd");
        thread_->join();
        thread_ = nullptr;
    }

    if (event_fd_ >= 0) close(event_fd_);
    if (epoll_fd_ >= 0) close(epoll_fd_);
    for (auto& it : pid_fds_) { close(it.second); }
    
    event_fd_ = -1;
    epoll_fd_ = -1;
    pid_fds_.clear();
}

void PidWaiter::AddWait(PID pid)
{
    int pid_fd = OpenPidFd(pid, 0);
    XASSERT(pid_fd >= 0, "fail to open pid fd");

    mtx_.lock();
    pid_fds_[pid] = pid_fd;
    mtx_.unlock();

    // According to the notes of linux man page of epoll_wait at
    // https://www.man7.org/linux/man-pages/man2/epoll_wait.2.html
    // "While one thread is blocked in a call to epoll_wait(), it is
    // possible for another thread to add a file descriptor to the
    // waited-upon epoll instance. If the new file descriptor becomes
    // ready, it will cause the epoll_wait() call to unblock."
    // So we can safely add the pid fd to epoll.
    struct epoll_event ev;
    ev.events = EPOLLIN;
    ev.data.u64 = PackEventData(kEpollEventPid, pid);
    XASSERT(!epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, pid_fd, &ev),
            "fail to add pid fd to epoll");
}

void PidWaiter::WaitWorker()
{
    struct epoll_event ev;
    while (true) {
        if (epoll_wait(epoll_fd_, &ev, 1, -1) == -1) {
            XASSERT(errno == EINTR, "fail during epoll wait");
			continue;
		}

        if (GetEventType(ev.data.u64) == kEpollEventTerminate) {
            eventfd_t v;
            XASSERT(!eventfd_read(event_fd_, &v), "fail to read event fd");
            return;
        }

        XASSERT(GetEventType(ev.data.u64) == kEpollEventPid,
                "invalid event type: %d", GetEventType(ev.data.u64));

        PID pid = GetEventPid(ev.data.u64);
        
        mtx_.lock();
        auto it = pid_fds_.find(pid);
        XASSERT(it != pid_fds_.end(), "pid fd not found");
        int pid_fd = it->second;
        pid_fds_.erase(it);
        mtx_.unlock();

        XASSERT(!epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, pid_fd, nullptr),
                "fail to remove pid fd from epoll");
        XASSERT(!close(pid_fd), "fail to close pid fd");
        callback_(pid);
    }
}

PID PidWaiter::GetEventPid(uint64_t data)
{
    return PID(data & 0xFFFFFFFF);
}

EpollEventType PidWaiter::GetEventType(uint64_t data)
{
    return EpollEventType(data >> 32);
}

uint64_t PidWaiter::PackEventData(EpollEventType type, PID pid)
{
    return ((uint64_t)type << 32) | (uint64_t)pid;
}
