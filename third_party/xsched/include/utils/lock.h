#pragma once

#include <mutex>
#include <atomic>
#include <pthread.h>

namespace xsched::utils
{

class MutexLock
{
public:
    MutexLock() = default;
    virtual ~MutexLock() = default;

    virtual void lock() = 0;
    virtual void unlock() = 0;
};

class StdMutex : public MutexLock
{
private:
    std::mutex mutex_;

public:
    StdMutex() = default;
    virtual ~StdMutex() = default;

    virtual void lock() override { mutex_.lock(); }
    virtual void unlock() override { mutex_.unlock(); }
};

class SpinLock : public MutexLock
{
public:
    SpinLock();
    virtual ~SpinLock();

    virtual void lock() override;
    virtual void unlock() override;

    void tryLock();

private:
    pthread_spinlock_t spinlock_;
};

class MCSLock : public MutexLock
{
public:
    MCSLock() = default;
    virtual ~MCSLock() = default;

    virtual void lock() override;
    virtual void unlock() override;

private:
    enum LockStatus
    {
        kLockWaiting = 0,
        kLockGranted = 1
    };

    struct MCSNode
    {
        volatile LockStatus flag;
        volatile MCSNode *next;
    } __attribute__((aligned(64)));

    static thread_local MCSNode me;
    std::atomic<MCSNode *> tail_{nullptr};
};

} // namespace xsched::utils
