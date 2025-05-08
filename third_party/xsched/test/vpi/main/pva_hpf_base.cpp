#include "test.h"
#include "blur.h"
#include "utils.h"
#include "utils/log.h"
#include "shim/vpi/xctrl.h"

#include <thread>
#include <vpi/Stream.h>

int main(int argc, char **argv)
{
    if (argc < 3) {
        XINFO("Usage: %s <input_video> <output_video>",
              argv[0]);
        XERRO("lack arguments, abort...");
    }

    PvaBlurRunner runner_rt(argv[1], argv[2]);
    PvaBlurRunner runner_be(argv[1], argv[2]);
    VPIStream stream_rt = runner_rt.CreateStream();
    VPIStream stream_be = runner_be.CreateStream();
    runner_rt.Init(stream_rt);
    runner_be.Init(stream_be);

    std::thread thd([&]() -> void {
        RtTask(&runner_rt, stream_rt, 30, 40, 20);
    });
    
    BeTask(&runner_be, stream_be, 30);

    thd.join();
}
