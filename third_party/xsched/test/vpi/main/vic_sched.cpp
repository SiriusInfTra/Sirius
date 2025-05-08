#include "test.h"
#include "utils.h"
#include "rescale.h"
#include "utils/log.h"
#include "shim/vpi/xctrl.h"

#include <vpi/Stream.h>

int main(int argc, char **argv)
{
    if (argc < 5) {
        XINFO("Usage: %s <input_video> <output_video> <w_out> <h_out>",
              argv[0]);
        XERRO("lack arguments, abort...");
    }

    VicRescaleRunner runner_rt(argv[1], argv[2],
        atoi(argv[3]), atoi(argv[4]));
    VicRescaleRunner runner_be(argv[1], argv[2],
        atoi(argv[3]), atoi(argv[4]));
    VPIStream stream_rt = runner_rt.CreateStream();
    VPIStream stream_be = runner_be.CreateStream();
    runner_rt.Init(stream_rt);
    runner_be.Init(stream_be);

    CoRun(&runner_rt, stream_rt, 15, &runner_be, stream_be, 15);

    uint64_t xq_rt = VpiXQueueCreate(stream_rt, PREEMPT_MODE_STOP_SUBMISSION,
                                     8, 4);
    uint64_t xq_be = VpiXQueueCreate(stream_be, PREEMPT_MODE_STOP_SUBMISSION,
                                     8, 4);

    WarmUp(&runner_rt, stream_rt, 65535);
    WarmUp(&runner_be, stream_be, 65535);

    VpiXQueueSetPriority(stream_rt, 2);
    VpiXQueueSetPriority(stream_be, 1);

    CoRun(&runner_rt, stream_rt, 65535, &runner_be, stream_be, 65535);

    VpiXQueueDestroy(xq_rt);
    VpiXQueueDestroy(xq_be);
}
