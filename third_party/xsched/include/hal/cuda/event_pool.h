#pragma once

#include <queue>
#include <mutex>

#include "utils/pool.h"
#include "utils/common.h"
#include "hal/cuda/cuda.h"
#include "hal/cuda/driver.h"
#include "hal/cuda/cuda_assert.h"

namespace xsched::hal::cuda
{

class EventPool : public xsched::utils::ObjectPool
{
public:
    EventPool() = default;
    virtual ~EventPool() = default;

private:
    virtual void *Create() override;
};

extern EventPool g_event_pool;

} // namespace xsched::hal::cuda
