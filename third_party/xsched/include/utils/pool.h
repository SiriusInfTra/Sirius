#pragma once

#include <mutex>
#include <vector>

namespace xsched::utils
{

class ObjectPool
{
public:
    ObjectPool() { pool_.reserve(kPoolSize); }
    virtual ~ObjectPool() = default;
    
    void *Pop();
    void Push(void *obj);

private:
    virtual void *Create() = 0; 

    std::mutex mutex_;
    std::vector<void *> pool_;
    static const size_t kPoolSize = 512;
};

} // namespace xsched::utils
