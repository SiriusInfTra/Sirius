#include <set>
#include <sstream>

#include "utils/xassert.h"
#include "service/cli/cli.h"
#include "preempt/xqueue/xtype.h"
#include "sched/policy/cbs.h"
#include "sched/protocol/event.h"
#include "sched/protocol/protocol.h"

using namespace tabulate;
using namespace xsched::utils;
using namespace xsched::sched;
using namespace xsched::preempt;
using namespace xsched::service;

Cli::Cli(const std::string& addr, uint16_t port)
    : kAddr(addr), kPort(port)
{
    client_ = std::make_unique<TcpClient>();
}

Cli::~Cli()
{
    if(client_) client_->Close();
}

int Cli::ListXQueues()
{
    Json::Value request;
    request["request_type"] = "query_xqueue_status";

    std::string str = Json::writeString(json_writer_, request);

    client_->Connect(kAddr, kPort);
    client_->Send(str.c_str(), str.size());

    std::vector<char> data;
    client_->Recv(data);
    client_->Close();

    Json::Value response;
    json_reader_.parse(data.data(),
                       data.data() + data.size(),
                       response, false);

    Table table;
    table.add_row({"Handle", "Device", "Process ID", "Status", "Schedule"});
    table.row(0).format().font_style({FontStyle::bold});

    if (response["xqueue_status"].isNull()) {
        std::cout << table << std::endl;
        return 0;
    }
    
    size_t row = 1;
    for (const auto &xqueue : response["xqueue_status"]) {
        table.add_row({
            ToHex(xqueue["handle"].asUInt64()),
            kDeviceMap.at((XDevice)xqueue["device"].asInt()),
            xqueue["pid"].asString(),
            xqueue["ready"].asBool() ? "Ready" : "Block",
            xqueue["suspended"].asBool() ? "Suspend" : "Execute"
        });

        if (xqueue["ready"].asBool()) {
            table[row][3].format().font_color(Color::cyan);
        } else {
            table[row][3].format().font_color(Color::yellow);
        }

        if (xqueue["suspended"].asBool()) {
            table[row][4].format().font_color(Color::red);
        } else {
            table[row][4].format().font_color(Color::green);
        }

        row++;
    }
    std::cout << table << std::endl;
    return 0;
}

int Cli::ListProcesses()
{
    Json::Value request;
    request["request_type"] = "query_xqueue_status";

    std::string str = Json::writeString(json_writer_, request);

    client_->Connect(kAddr, kPort);
    client_->Send(str.c_str(), str.size());

    std::vector<char> data;
    client_->Recv(data);
    client_->Close();

    Json::Value response;
    json_reader_.parse(data.data(),
                       data.data() + data.size(),
                       response, false);

    Table table;
    table.add_row({"Process ID", "Devices", "XQueues"});
    table.row(0).format().font_style({FontStyle::bold});

    if (response["xqueue_status"].isNull()) {
        std::cout << table << std::endl;
        return 0;
    }

    struct ProcessStatus
    {
        PID pid;
        std::set<XDevice> devices;
        size_t xqueues;
    };

    std::map<PID, ProcessStatus> process_map;
    for (const auto &xqueue : response["xqueue_status"]) {
        PID pid = xqueue["pid"].asUInt();
        if (process_map.find(pid) == process_map.end()) {
            process_map[pid] = {pid, {(XDevice)xqueue["device"].asInt()}, 1};
            continue;
        }

        process_map[pid].devices.insert((XDevice)xqueue["device"].asInt());
        process_map[pid].xqueues++;
    }

    for (const auto &process : process_map) {
        std::string devices;
        for (const auto &device : process.second.devices) {
            devices += kDeviceMap.at(device) + "/";
        }
        table.add_row({
            std::to_string(process.second.pid),
            devices.substr(0, devices.size() - 1),
            std::to_string(process.second.xqueues)
        });
    }
    std::cout << table << std::endl;
    return 0;
}

int Cli::Top(uint64_t interval_ms)
{
    while (true) {
        std::cout << "\033[2J\033[H";
        ListXQueues();
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }
    return 0;
}

int Cli::SetPriority(XQueueHandle handle, Prio priority)
{
    auto hint = std::make_unique<SetPriorityHint>(priority, handle);
    SendHint(std::move(hint));
    std::cout << "priority set to " << priority << " for XQueue "
              << ToHex(handle) << std::endl;
    return 0;
}

int Cli::SetBandwidth(XQueueHandle handle, Bwidth bandwidth)
{
    auto hint = std::make_unique<SetBandwidthHint>(bandwidth, handle);
    SendHint(std::move(hint));
    std::cout << "bandwidth set to " << bandwidth << " for XQueue "
              << ToHex(handle) << std::endl;
    return 0;
}

int Cli::SetTimeslice(uint64_t timeslice_us)
{
    auto hint = std::make_unique<SetTimesliceHint>(timeslice_us);
    SendHint(std::move(hint));
    std::cout << "timeslice set to " << timeslice_us << " us" << std::endl;
    return 0;
}

std::string Cli::ToHex(uint64_t x)
{
    std::stringstream ss;
    ss << "0x" << std::hex << x;
    return ss.str();
}

void Cli::SendHint(std::unique_ptr<const Hint> hint)
{
    std::string server_name = kServerChannelName;
    auto send_chan = std::make_unique<ipc::channel>(server_name.c_str(),
                                                    ipc::sender);
    auto event = std::make_unique<HintEvent>(std::move(hint));
    XASSERT(send_chan->send(event->Data(), event->Size()),
            "Cannot send Hint type(%d) to server. Is the server running?",
            event->Type());
    send_chan->disconnect();
}
