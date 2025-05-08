#pragma once

#include <memory>
#include <thread>

#include "sched/protocol/hint.h"
#include "sched/protocol/event.h"
#include "sched/scheduler/scheduler.h"

namespace xsched::shim::xdag
{

class XDagAgent
{
public:
    XDagAgent();
    ~XDagAgent() = default;

    void Init();
    void GiveHint(std::unique_ptr<const sched::Hint> hint);
    void RecvEvent(std::unique_ptr<const sched::Event> event);
    void Execute(std::unique_ptr<const sched::Operation> operation);

private:
    std::unique_ptr<sched::Scheduler> scheduler_ = nullptr;
};

extern XDagAgent g_xdag_agent;

} // namespace xsched::shim::xdag
