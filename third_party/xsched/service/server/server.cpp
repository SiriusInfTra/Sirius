#include "utils/log.h"
#include "utils/xassert.h"
#include "service/server/server.h"
#include "sched/protocol/protocol.h"

using namespace xsched::utils;
using namespace xsched::sched;
using namespace xsched::service;

Server::Server(const std::string &policy_name, const std::string &port)
{
    port_ = std::stoi(port);

    auto it = kPolicyTypeMap.find(policy_name);
    if (it == kPolicyTypeMap.end()) {
        XERRO("invalid policy name %s", policy_name.c_str());
    }
    XASSERT(it->second > kPolicyInternalMax,
            "server policy must be a customized policy");

    scheduler_ = std::make_unique<LocalScheduler>(it->second);
    XINFO("scheduler created with policy %s", policy_name.c_str());
}

Server::~Server()
{
    Stop();
}

void Server::Run()
{
    std::string server_name = kServerChannelName;
    recv_chan_ = std::make_unique<ipc::channel>(server_name.c_str(),
                                                ipc::receiver);
    self_chan_ = std::make_unique<ipc::channel>(server_name.c_str(),
                                                ipc::sender);

    scheduler_->SetExecutor(std::bind(&Server::Execute,
                                      this,
                                      std::placeholders::_1));
    scheduler_->Run();

    pid_waiter_ = std::make_unique<PidWaiter>(
        std::bind(&Server::ProcessTerminate,
                  this,
                  std::placeholders::_1));
    pid_waiter_->Start();

    tcp_server_ = std::make_unique<TcpServer>();
    tcp_server_->Run(port_, std::bind(&Server::TcpHandler,
                                      this,
                                      std::placeholders::_1,
                                      std::placeholders::_2,
                                      std::placeholders::_3));

    this->RecvWorker();
}

void Server::Stop()
{
    if (self_chan_) {
        auto e = std::make_unique<TerminateEvent>();
        XASSERT(self_chan_->send(e->Data(), e->Size()),
                "cannot send terminate event");
    }

    if (tcp_server_) {
        tcp_server_->Stop();
    }

    if (pid_waiter_) {
        pid_waiter_->Stop();
    }

    tcp_server_ = nullptr;
    pid_waiter_ = nullptr;
}

void Server::RecvWorker()
{
    XINFO("recv worker started");

    while (true) {
        auto data = recv_chan_->recv();
        auto e = Event::CopyConstructor(data.data());

        switch (e->Type())
        {
        case kEventTerminate:
            for (auto &it : client_chans_) {
                it.second->disconnect();
            }
            recv_chan_->disconnect();
            self_chan_->disconnect();
            scheduler_->Stop();

            client_chans_.clear();
            scheduler_ = nullptr;
            recv_chan_ = nullptr;
            self_chan_ = nullptr;
            return;
        case kEventProcessCreate:
        {
            PID client_pid = e->Pid();
            std::string client_name = kClientChannelPrefix
                                    + std::to_string(client_pid);
            auto client_chan = std::make_shared<ipc::channel>(
                client_name.c_str(), ipc::sender);

            XINFO("client process (%u) connected", client_pid);

            chan_mtx_.lock();
            client_chans_[client_pid] = client_chan;
            chan_mtx_.unlock();
            scheduler_->RecvEvent(std::move(e));
            pid_waiter_->AddWait(client_pid);
            break;
        }
        case kEventProcessDestroy:
        {
            PID client_pid = e->Pid();
            chan_mtx_.lock();
            auto it = client_chans_.find(client_pid);
            bool closed = it == client_chans_.end();
            chan_mtx_.unlock();

            if (closed) break;

            scheduler_->RecvEvent(std::move(e));
            chan_mtx_.lock();
            client_chans_.erase(client_pid);
            chan_mtx_.unlock();

            XINFO("client process (%u) closed", client_pid);
            break;
        }
        default:
            scheduler_->RecvEvent(std::move(e));
            break;
        }
    }
}

void Server::ProcessTerminate(PID pid)
{
    auto e = std::make_unique<ProcessDestroyEvent>(pid);
    XASSERT(self_chan_->send(e->Data(), e->Size()),
            "cannot send process destory event");
}

void Server::QueryXQueueStatus(Json::Value &response)
{
    static StatusQuery query;
    query.Reset();
    query.status_.reserve(128);
    
    auto e = std::make_unique<StatusQueryEvent>(&query);
    XASSERT(self_chan_->send(e->Data(), e->Size()),
            "cannot send status query event");
    query.Wait();

    // TODO: Process the status query result.
    Json::Value status;
    for (const auto &s : query.status_) {
        Json::Value xqueue;
        xqueue["handle"] = (Json::UInt64)s->handle;
        xqueue["device"] = (Json::Int)s->device;
        xqueue["pid"] = (Json::UInt)s->pid;
        xqueue["ready"] = s->ready;
        xqueue["suspended"] = s->suspended;
        status.append(xqueue);
    }
    response["xqueue_status"] = status;
}

void Server::Execute(std::unique_ptr<const Operation> operation)
{
    PID client_pid = operation->Pid();

    chan_mtx_.lock();
    auto it = client_chans_.find(client_pid);

    if (it == client_chans_.end()) {
        chan_mtx_.unlock();
        XWARN("cannot find client channel for client process %u",
              client_pid);
        return;
    }

    std::shared_ptr<ipc::channel> client_chan = it->second;
    chan_mtx_.unlock();

    XASSERT(client_chan->send(operation->Data(), operation->Size()),
            "cannot send operation to client process %u", client_pid);
}

void Server::TcpHandler(const void *data, size_t size,
                        std::vector<char> &response_data)
{
    Json::Value request;

    if (!json_reader_.parse((const char *)data,
                            (const char *)data + size,
                            request, false)) {
        ((char *)data)[size] = '\0';
        XWARN("fail to parse request: %s", (const char *)data);
    }

    if (!request.isMember("request_type")) {
        XWARN("request field 'request_type' missing");
        return;
    }

    std::string request_type = request["request_type"].asString();
    auto it = kRequestTypeMap.find(request_type);
    if (it == kRequestTypeMap.end()) {
        XWARN("invalid request type %s", request_type.c_str());
        return;
    }

    Json::Value response;
    switch (it->second)
    {
    case kRequestTypeQueryXQueueStatus:
        QueryXQueueStatus(response);
        break;
    default:
        XWARN("request type %s not supported", request_type.c_str());
        return;
    }

    std::string str = Json::writeString(json_writer_, response);

    response_data.resize(str.size());
    std::copy(str.begin(), str.end(), response_data.begin());
}
