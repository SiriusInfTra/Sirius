#include "test.h"
#include "blur.h"
#include "utils.h"
#include "utils/log.h"
#include "shim/vpi/xctrl.h"

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

    uint64_t xq_rt = VpiXQueueCreate(stream_rt, PREEMPT_MODE_STOP_SUBMISSION,
                                     32, 16);
    uint64_t xq_be = VpiXQueueCreate(stream_be, PREEMPT_MODE_STOP_SUBMISSION,
                                     16, 8);

    WarmUp(&runner_rt, stream_rt, 30);
    WarmUp(&runner_be, stream_be, 30);

    VpiXQueueSetPriority(stream_rt, 2);
    VpiXQueueSetPriority(stream_be, 1);

    HpfCoRun(&runner_rt, stream_rt, 30,
             &runner_be, stream_be, 30,
             40, 20);

    VpiXQueueDestroy(xq_rt);
    VpiXQueueDestroy(xq_be);
}
