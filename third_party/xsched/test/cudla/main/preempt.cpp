#include "test.h"
#include "utils.h"
#include "model.h"
#include "utils/log.h"
#include "shim/cudla/xctrl.h"

#include <cuda_runtime.h>

int main(int argc, char **argv)
{
	if (argc < 2) {
        XINFO("Usage: %s <engine dir>", argv[0]);
        XERRO("lack arguments, abort...");
    }

	CudlaModel model(argv[1]);
    cudaStream_t stream;
    CUDART_ASSERT(cudaStreamCreate(&stream));

    XINFO("[TEST] inference baseline test");
    WarmUp(model, stream);
    int64_t disable_ns = InferLatencyNs(model, stream);
    XINFO("[RESULT] disable preemption: %ldus", disable_ns / 1000);

    XINFO("[TEST] inference overhead test");
    XINFO("[TEST] CudlaXQueueCreate()");
    uint64_t handle = CudlaXQueueCreate(stream, PREEMPT_MODE_STOP_SUBMISSION,
                                        2, 1);

    WarmUp(model, stream);
    int64_t enable_ns = InferLatencyNs(model, stream);
    XINFO("[RESULT] enable preemption: %ldus", enable_ns / 1000);
    XINFO("[RESULT] overhead: %ldus, %.2f%%",
          (enable_ns - disable_ns) / 1000,
          double(enable_ns - disable_ns) / double(disable_ns) * 100);

    XINFO("[TEST] inference preempt test");
    PreemptTest(model, stream);

    XINFO("[TEST] test complete");
    XINFO("[TEST] CudlaXQueueDestroy()");
    CudlaXQueueDestroy(handle);
    CUDART_ASSERT(cudaStreamDestroy(stream));
    return 0;
}
