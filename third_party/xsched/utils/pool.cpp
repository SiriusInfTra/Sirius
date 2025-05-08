#include "utils/pool.h"

using namespace xsched::utils;

void *ObjectPool::Pop()
{
    void *obj;
    mutex_.lock();

    if (pool_.empty()) {
        for (size_t i = 0; i < kPoolSize; ++i) {
            obj = Create();
            pool_.emplace_back(obj);
        }
        obj = Create();
    } else {
        obj = pool_.back();
        pool_.pop_back();
    }

    mutex_.unlock();
    return obj;
}

void ObjectPool::Push(void *obj)
{
    mutex_.lock();
    pool_.emplace_back(obj);
    mutex_.unlock();
}
