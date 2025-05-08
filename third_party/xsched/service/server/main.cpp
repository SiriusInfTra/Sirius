#include <memory>
#include <csignal>

#include "utils/log.h"
#include "utils/xassert.h"
#include "service/server/server.h"

using namespace xsched::service;

static std::unique_ptr<Server> server = nullptr;

void ExitSignal(int)
{
    if (server) server->Stop();
}

int main(int argc, char *argv[])
{
    if (argc < 3) {
        XINFO("usage: %s <policy name> <port number>", argv[0]);
        XERRO("lack arguments, abort...");
    }

    XASSERT(signal(SIGINT, ExitSignal) != SIG_ERR,
            "failed to set SIGINT handler");
    XASSERT(signal(SIGQUIT, ExitSignal) != SIG_ERR,
            "failed to set SIGQUIT handler");

    server = std::make_unique<Server>(argv[1], argv[2]);
    server->Run();
    return 0;
}
