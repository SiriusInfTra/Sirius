#include "utils/log.h"
#include "utils/common.h"
#include "cuxtra/cuxtra.h"
#include "shim/xdag/xctrl.h"
#include "shim/xdag/agent.h"
#include "shim/xdag/handle.h"
#include "sched/scheduler/local.h"
#include "sched/scheduler/global.h"
#include "sched/protocol/protocol.h"

using namespace xsched::sched;
using namespace xsched::preempt;

xsched::shim::xdag::XDagAgent xsched::shim::xdag::g_xdag_agent;

namespace xsched::shim::xdag
{

XDagAgent::XDagAgent()
{
    this->Init();
}

void XDagAgent::Init()
{
    if (scheduler_ != nullptr) return;

    scheduler_ = std::make_unique<LocalScheduler>(kPolicyHighestPriorityFirst);
    scheduler_->SetExecutor(std::bind(&XDagAgent::Execute,
                                      this,
                                      std::placeholders::_1));
    scheduler_->Run();
}

void XDagAgent::GiveHint(std::unique_ptr<const sched::Hint> hint)
{
    if (scheduler_ == nullptr) {
        XWARN("Scheduler not initialized, Hint type(%d) dropped",
              hint->Type());
        return;
    }

    auto event = std::make_unique<HintEvent>(std::move(hint));
    scheduler_->RecvEvent(std::move(event));
}

void XDagAgent::RecvEvent(std::unique_ptr<const Event> event)
{
    if (scheduler_ == nullptr) {
        XWARN("Scheduler not initialized, Event type(%d) dropped",
              event->Type());
        return;
    }
    scheduler_->RecvEvent(std::move(event));
}

void XDagAgent::Execute(std::unique_ptr<const Operation> operation)
{
    if (operation->Type() != kOperationSched) return;

    const SchedOperation *sched_op = (const SchedOperation *)operation.get();
    size_t running_cnt = sched_op->RunningCnt();
    size_t suspended_cnt = sched_op->SuspendedCnt();
    const XQueueHandle *handles = sched_op->Handles();

    for (size_t i = 0; i < running_cnt; ++i) {
        XDagResume(handles[i]);
    }
    for (size_t i = 0; i < suspended_cnt; ++i) {
        XDagSuspend(handles[running_cnt + i]);
    }
}

} // namespace xsched::shim::xdag
