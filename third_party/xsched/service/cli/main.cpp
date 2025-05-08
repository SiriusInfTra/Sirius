#include <memory>
#include <iostream>
#include <CLI/CLI11.hpp>

#include "service/cli/cli.h"

using namespace xsched::service;

int main(int argc, char **argv)
{
    auto xcli = std::make_unique<CLI::App>("App description");
    
    std::string addr = "127.0.0.1";
    xcli->add_option("-a,--addr", addr, "XSched server ip address")
        ->check(CLI::ValidIPV4);

    uint16_t port = 50000;
    xcli->add_option("-p,--port", port, "XSched server port")
        ->check(CLI::Range(0x1U, 0xFFFFU));
    
    // subcommand top
    CLI::App *top = xcli->add_subcommand("top",
        "Show the top information of the XQueues");
    uint64_t interval_ms = 1000;
    top->add_option("-i,--interval", interval_ms,
                    "Interval of refreshing the information in milliseconds. "
                    "Default is 1000 ms.")
       ->check(CLI::Range(10ULL, 2000ULL));

    // subcommand list
    CLI::App *list = xcli->add_subcommand("list",
        "List information about one kind of objects");
    std::string object;
    list->add_option("object", object, "Object to list")
        ->check(CLI::IsMember({ "xqueue", "process" }));
    
    // subcommand hint
    CLI::App *hint = xcli->add_subcommand("hint",
        "Give a hint to the XSched server");
    
    xsched::preempt::XQueueHandle handle = 0;

    // subcommand hint priority
    CLI::App *set_priority_hint = hint->add_subcommand("priority",
        "Set the priority of the XQueue");
    set_priority_hint->add_option("handle", handle,
                                  "Handle of the XQueue")
                     ->required();
    xsched::sched::Prio priority = 0;
    set_priority_hint->add_option("prio", priority,
                                  "Priority of the XQueue. "
                                  "Higher value means higher priority.")
                     ->check(CLI::Range(PRIO_MIN, PRIO_MAX))
                     ->required();
    
    // subcommand hint bandwidth
    CLI::App *set_bandwidth_hint = hint->add_subcommand("bandwidth",
        "Set the bandwidth of the XQueue");
    set_bandwidth_hint->add_option("handle", handle,
                                   "Handle of the XQueue")
                      ->required();
    xsched::sched::Bwidth bandwidth = 0;
    set_bandwidth_hint->add_option("bwidth", bandwidth,
                                   "Bandwidth of the XQueue. "
                                   "Higher value means higher bandwidth.")
                      ->check(CLI::Range(BANDWIDTH_MIN, BANDWIDTH_MAX))
                      ->required();

    // subcommand hint timeslice
    CLI::App *set_timeslice_hint = hint->add_subcommand("timeslice",
        "Set the timeslice of the CBS scheduler");
    uint64_t timeslice_us = 0x2000;
    set_timeslice_hint->add_option("us", timeslice_us,
                                   "Timeslice in microseconds. "
                                   "Default is 8192 us.")
                      ->check(CLI::Range(0x400ULL, 0x10000ULL))
                      ->required();

    CLI11_PARSE(*xcli, argc, argv);

    Cli cli(addr, port);

    if (top->parsed()) return cli.Top(interval_ms);

    if (list->parsed()) {
        if (object == "xqueue") return cli.ListXQueues();
        if (object == "process") return cli.ListProcesses();
    }

    if (hint->parsed()) {
        if (set_priority_hint->parsed()) return cli.SetPriority(handle, priority);
        if (set_bandwidth_hint->parsed()) return cli.SetBandwidth(handle, bandwidth);
        if (set_timeslice_hint->parsed()) return cli.SetTimeslice(timeslice_us);
    }

    std::cerr << "Unknown command" << std::endl;
    return 1;
}
