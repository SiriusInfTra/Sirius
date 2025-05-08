#include "utils/log.h"
#include "utils/common.h"
#include "shim/common/agent.h"
#include "shim/common/xmanager.h"
#include "sched/protocol/protocol.h"
#include "preempt/event/dispatcher.h"

using namespace xsched::sched;
using namespace xsched::preempt;

xsched::shim::SchedAgent xsched::shim::g_sched_agent;

namespace xsched::shim
{

SchedAgent::SchedAgent()
{
    this->Init();
}

void SchedAgent::Init()
{
    if (scheduler_ != nullptr) return;
    
    scheduler_ = CreateScheduler();
    scheduler_->SetExecutor(std::bind(&SchedAgent::Execute,
                                      this,
                                      std::placeholders::_1));
    scheduler_->Run();
    g_event_dispatcher.AddListener(std::bind(&SchedAgent::RecvEvent,
                                             this,
                                             std::placeholders::_1));
}

void SchedAgent::GiveHint(std::unique_ptr<const sched::Hint> hint)
{
    if (scheduler_ == nullptr) {
        XWARN("Scheduler not initialized, Hint type(%d) dropped",
              hint->Type());
        return;
    }

    auto event = std::make_unique<HintEvent>(std::move(hint));
    scheduler_->RecvEvent(std::move(event));
}

void SchedAgent::RecvEvent(std::unique_ptr<const Event> event)
{
    if (scheduler_ == nullptr) {
        XWARN("Scheduler not initialized, Event type(%d) dropped",
              event->Type());
        return;
    }
    scheduler_->RecvEvent(std::move(event));
}

void SchedAgent::Execute(std::unique_ptr<const Operation> operation)
{
    if (operation->Type() != kOperationSched) return;

    const SchedOperation *sched_op = (const SchedOperation *)operation.get();
    size_t running_cnt = sched_op->RunningCnt();
    size_t suspended_cnt = sched_op->SuspendedCnt();
    const XQueueHandle *handles = sched_op->Handles();

    for (size_t i = 0; i < running_cnt; ++i) {
        auto xqueue = XManager::GetXQueue(handles[i]);
        if (xqueue) xqueue->Resume(false);
    }
    for (size_t i = 0; i < suspended_cnt; ++i) {
        auto xqueue = XManager::GetXQueue(handles[running_cnt + i]);
        if (xqueue) xqueue->Suspend(false);
    }
}

} // namespace xsched::shim
