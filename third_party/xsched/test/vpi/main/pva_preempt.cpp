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

    PvaBlurRunner runner(argv[1], argv[2]);
    VPIStream stream = runner.CreateStream();
    runner.Init(stream);

    XINFO("[TEST] execute baseline test");
    int64_t disable_ns = ExecuteLatencyNs(&runner, stream, 30, true);
    XINFO("[RESULT] disable preemption sync: %ldus", disable_ns / 1000);

    disable_ns = ExecuteLatencyNs(&runner, stream, 30, false);
    XINFO("[RESULT] disable preemption async: %ldus", disable_ns / 1000);

    XINFO("[TEST] execute overhead test");
    XINFO("[TEST] VpiXQueueCreate()");
    uint64_t handle = VpiXQueueCreate(stream, PREEMPT_MODE_STOP_SUBMISSION,
                                      8, 4);

    WarmUp(&runner, stream, 65535);
    int64_t enable_ns = ExecuteLatencyNs(&runner, stream, 65535, false);
    XINFO("[RESULT] enable preemption: %ldus", enable_ns / 1000);
    XINFO("[RESULT] overhead: %ldus, %.2f%%",
          (enable_ns - disable_ns) / 1000,
          double(enable_ns - disable_ns) / double(disable_ns) * 100);

    XINFO("[TEST] preempt test");
    PreemptTest(&runner, stream, 65535);

    XINFO("[TEST] test complete");
    runner.Final(stream);

    XINFO("[TEST] VpiXQueueDestroy()");
    VpiXQueueDestroy(handle);

    vpiStreamDestroy(stream);
    return 0;
}
