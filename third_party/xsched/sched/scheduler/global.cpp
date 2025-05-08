#include "utils/xassert.h"
#include "sched/scheduler/global.h"
#include "sched/protocol/protocol.h"

using namespace xsched::sched;

GlobalScheduler::GlobalScheduler()
    : Scheduler(kSchedulerGlobal)
{

}

GlobalScheduler::~GlobalScheduler()
{
    Stop();
}

void GlobalScheduler::Run()
{
    std::string server_name = kServerChannelName;
    client_chan_name_ = kClientChannelPrefix + std::to_string(GetProcessId());

    recv_chan_ = std::make_unique<ipc::channel>(client_chan_name_.c_str(),
                                                ipc::receiver);
    send_chan_ = std::make_unique<ipc::channel>(server_name.c_str(),
                                                ipc::sender);

    auto event = std::make_unique<ProcessCreateEvent>();
    XASSERT(send_chan_->send(event->Data(), event->Size()),
            "cannot send ProcessCreateEvent to server "
            "Is the server running?");
    
    thread_ = std::make_unique<std::thread>(&GlobalScheduler::Worker, this);
}

void GlobalScheduler::Stop()
{
    if (thread_) {
        auto event = std::make_unique<ProcessDestroyEvent>();
        XASSERT(send_chan_->send(event->Data(), event->Size()),
                "cannot send ProcessDestroyEvent to server");

        auto op = std::make_unique<TerminateOperation>();
        auto self_chan = std::make_unique<ipc::channel>(
            client_chan_name_.c_str(), ipc::sender);
        XASSERT(self_chan->send(op->Data(), op->Size()),
                "cannot send TerminateOperation to worker thread");
        thread_->join();
    }

    thread_ = nullptr;
    recv_chan_ = nullptr;
    send_chan_ = nullptr;
}

void GlobalScheduler::RecvEvent(std::unique_ptr<const Event> event)
{
    XASSERT(send_chan_->send(event->Data(), event->Size()),
            "cannot send event to server");
}

void GlobalScheduler::Worker()
{
    while (true) {
        auto data = recv_chan_->recv();
        auto op = Operation::CopyConstructor(data.data());
        if (UNLIKELY(op->Type() == kOperationTerminate)) return;
        Execute(std::move(op));
    }
}
