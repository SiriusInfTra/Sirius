#pragma once

#include <queue>
#include <mutex>

#include "utils/pool.h"
#include "utils/common.h"
#include "hal/vpi/driver.h"
#include "hal/vpi/vpi_assert.h"

namespace xsched::hal::vpi
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

} // namespace xsched::hal::vpi
