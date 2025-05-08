#pragma once

#include <mutex>
#include <memory>
#include <thread>
#include <string>
#include <unordered_map>
#include <libipc/ipc.h>
#include <jsoncpp/json/json.h>

#include "utils/tcp.h"
#include "utils/waitpid.h"
#include "sched/scheduler/local.h"

namespace xsched::service
{

class Server
{
public:
    Server(const std::string &policy_name, const std::string &port);
    ~Server();

    void Run();
    void Stop();

private:
    void RecvWorker();
    void ProcessTerminate(PID pid);
    void QueryXQueueStatus(Json::Value &response);
    void Execute(std::unique_ptr<const sched::Operation> operation);
    void TcpHandler(const void *data, size_t size,
                    std::vector<char> &response);

    std::mutex chan_mtx_;
    std::unique_ptr<ipc::channel> recv_chan_ = nullptr;
    std::unique_ptr<ipc::channel> self_chan_ = nullptr;
    std::unordered_map<PID, std::shared_ptr<ipc::channel>> client_chans_;

    std::unique_ptr<utils::PidWaiter> pid_waiter_ = nullptr;
    
    uint16_t port_ = 0;
    Json::Reader json_reader_;
    Json::StreamWriterBuilder json_writer_;
    std::unique_ptr<utils::TcpServer> tcp_server_ = nullptr;

    std::unique_ptr<sched::LocalScheduler> scheduler_ = nullptr;
};

} // namespace xsched::service
