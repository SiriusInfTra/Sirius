#pragma once

#include <memory>
#include <libipc/ipc.h>
#include <tabulate/table.hpp>
#include <jsoncpp/json/json.h>

#include "utils/tcp.h"
#include "sched/policy/hpf.h"
#include "sched/policy/cbs.h"
#include "sched/protocol/hint.h"
#include "preempt/xqueue/xtype.h"

namespace xsched::service
{

class Cli
{
public:
    Cli(const std::string& addr, uint16_t port);
    ~Cli();

    int ListXQueues();
    int ListProcesses();
    int Top(uint64_t interval_ms);

    // hint related apis
    int SetPriority(preempt::XQueueHandle handle, sched::Prio priority);
    int SetBandwidth(preempt::XQueueHandle handle, sched::Bwidth bandwidth);
    int SetTimeslice(uint64_t timeslice_us);

private:
    std::string ToHex(uint64_t x);
    void SendHint(std::unique_ptr<const sched::Hint> hint);

    const std::string kAddr;
    const uint16_t kPort;

    Json::Reader json_reader_;
    Json::StreamWriterBuilder json_writer_;
    std::unique_ptr<utils::TcpClient> client_ = nullptr;
};

};
