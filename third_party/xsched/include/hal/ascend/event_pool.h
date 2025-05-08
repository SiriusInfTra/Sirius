#pragma once

#include "utils/pool.h"

namespace xsched::hal::ascend
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

} // namespace xsched::hal::ascend
