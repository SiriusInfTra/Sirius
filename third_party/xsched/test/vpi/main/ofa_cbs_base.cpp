#include "sde.h"
#include "test.h"
#include "utils.h"
#include "utils/log.h"
#include "shim/vpi/xctrl.h"

#include <thread>
#include <vpi/Stream.h>

int main(int argc, char **argv)
{
    if (argc < 3) {
        XINFO("Usage: %s <left image> <right image>",
              argv[0]);
        XERRO("lack arguments, abort...");
    }

    OfaSdeRunner runner1(argv[1], argv[2]);
    OfaSdeRunner runner2(argv[1], argv[2]);
    VPIStream stream1 = runner1.CreateStream();
    VPIStream stream2 = runner2.CreateStream();
    runner1.Init(stream1);
    runner2.Init(stream2);

    std::thread thd([&]() -> void {
        CbsTask(&runner1, stream1, 8, "client1");
    });
    
    CbsTask(&runner2, stream2, 8, "client2");

    thd.join();
}
