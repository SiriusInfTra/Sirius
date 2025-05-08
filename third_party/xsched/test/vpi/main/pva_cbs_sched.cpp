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

    PvaBlurRunner runner1(argv[1], argv[2]);
    PvaBlurRunner runner2(argv[1], argv[2]);
    VPIStream stream1 = runner1.CreateStream();
    VPIStream stream2 = runner2.CreateStream();
    runner1.Init(stream1);
    runner2.Init(stream2);

    uint64_t xq1 = VpiXQueueCreate(stream1, PREEMPT_MODE_STOP_SUBMISSION,
                                   2, 2);
    uint64_t xq2 = VpiXQueueCreate(stream2, PREEMPT_MODE_STOP_SUBMISSION,
                                   2, 2);

    WarmUp(&runner1, stream1, 8);
    WarmUp(&runner2, stream2, 8);

    VpiXQueueSetBandwidth(stream1, 3);
    VpiXQueueSetBandwidth(stream2, 1);

    CbsCoRun(&runner1, stream1, 8, &runner2, stream2, 8);

    VpiXQueueDestroy(xq1);
    VpiXQueueDestroy(xq2);
}
