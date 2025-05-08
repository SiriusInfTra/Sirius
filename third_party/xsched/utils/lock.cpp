#include "utils/lock.h"

#ifdef __x86_64__
    #define memory_barrier() asm volatile("pause" ::: "memory")
#elif __aarch64__
    #define memory_barrier() asm volatile("yield" ::: "memory")
#endif

using namespace xsched::utils;

SpinLock::SpinLock()
{
    pthread_spin_init(&spinlock_, PTHREAD_PROCESS_PRIVATE);
}

SpinLock::~SpinLock()
{
    pthread_spin_destroy(&spinlock_);
}

void SpinLock::lock()
{
    pthread_spin_lock(&spinlock_);
}

void SpinLock::unlock()
{
    pthread_spin_unlock(&spinlock_);
}

void SpinLock::tryLock()
{
    pthread_spin_trylock(&spinlock_);
}

thread_local MCSLock::MCSNode MCSLock::me;

void MCSLock::lock()
{
    MCSNode *tail = nullptr;
    me.flag = kLockWaiting;
    me.next = nullptr;
    tail = tail_.exchange(&me);
    if (tail) {
        tail->next = &me;
        while (me.flag != kLockGranted) {
            memory_barrier();
        }
    }
}

void MCSLock::unlock()
{
    if (!me.next) {
        MCSNode *me_ptr = &me;
        if (tail_.compare_exchange_strong(me_ptr, nullptr)) {
            return;
        }
        while (!me.next) {
            memory_barrier();
        }
    }
    me.next->flag = kLockGranted;
}
