#pragma once

#include <queue>
#include <mutex>

#include "utils/pool.h"
#include "utils/common.h"
#include "hal/cudla/driver.h"
#include "hal/cudla/cudla_assert.h"

namespace xsched::hal::cudla
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

} // namespace xsched::hal::cudla
