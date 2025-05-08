#include "test.h"
#include "utils.h"
#include "model.h"
#include "utils/log.h"
#include "utils/data.h"
#include "utils/timing.h"
#include "utils/runner.h"
#include "shim/cudla/xctrl.h"

#include <thread>
#include <cuda_runtime.h>

using namespace xsched::utils;

static void CoRun(CudlaModel &model_rt, cudaStream_t stream_rt,
                  CudlaModel &model_be, cudaStream_t stream_be)
{
    srand(time(NULL));

    LoopRunner runner;
    runner.Start([&]() -> void {
        model_be.Infer(stream_be);
    });

    DataProcessor<int64_t> latency;
    std::this_thread::sleep_for(std::chrono::seconds(2));

    for (int i = 0; i < LATENCY_TEST_CNT; ++i) {
        latency.Add(EXEC_TIME(nanoseconds, { model_rt.Infer(stream_rt); }));
        std::this_thread::sleep_for(std::chrono::microseconds(
            SLEEP_US / 2 + rand() % SLEEP_US));
    }
    runner.Stop();

    XINFO("[RESULT] latency avg %.2f us, p50 %.2f us, p90 %.2f us, p99 %.2f us",
          (double)latency.Avg() / 1000,
          (double)latency.Percentile(0.5) / 1000,
          (double)latency.Percentile(0.9) / 1000,
          (double)latency.Percentile(0.99) / 1000);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        XINFO("Usage: %s <engine dir>", argv[0]);
        XERRO("lack arguments, abort...");
    }

    CudlaModel model_rt(argv[1]);
	CudlaModel model_be(argv[1]);

    cudaStream_t stream_rt;
    cudaStream_t stream_be;
    CUDART_ASSERT(cudaStreamCreate(&stream_rt));
    CUDART_ASSERT(cudaStreamCreate(&stream_be));

    CoRun(model_rt, stream_rt, model_be, stream_be);

    uint64_t xq_rt = CudlaXQueueCreate(stream_rt, PREEMPT_MODE_STOP_SUBMISSION,
                                       2, 1);
    uint64_t xq_be = CudlaXQueueCreate(stream_be, PREEMPT_MODE_STOP_SUBMISSION,
                                       2, 1);

    WarmUp(model_rt, stream_rt);
    WarmUp(model_be, stream_be);

    CudlaXQueueSetPriority(stream_rt, 2);
    CudlaXQueueSetPriority(stream_be, 1);

    CoRun(model_rt, stream_rt, model_be, stream_be);

    CudlaXQueueDestroy(xq_rt);
    CudlaXQueueDestroy(xq_be);
}
