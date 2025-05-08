#include "test.h"
#include "utils.h"
#include "model.h"
#include "utils/log.h"
#include "utils/data.h"
#include "utils/psync.h"
#include "utils/timing.h"
#include "utils/runner.h"
#include "shim/cudla/xctrl.h"

#include <thread>
#include <cuda_runtime.h>

using namespace xsched::utils;

static void CoRun(CudlaModel &model1, cudaStream_t stream1,
                  CudlaModel &model2, cudaStream_t stream2)
{
    DataProcessor<int64_t> latency1;
    DataProcessor<int64_t> latency2;

    bool begin = false;

    LoopRunner runner;
    runner.Start([&]() -> void {
        int64_t exec_time = EXEC_TIME(nanoseconds, {
            model2.Infer(stream2);
        });
        if (begin) latency2.Add(exec_time);
    });

    std::this_thread::sleep_for(std::chrono::seconds(2));

    begin = true;
    double test_time = EXEC_TIME(nanoseconds, {
        for (int i = 0; i < LATENCY_TEST_CNT; ++i) {
            latency1.Add(EXEC_TIME(nanoseconds, {
                model1.Infer(stream1);
            }));
        }
    });
    runner.Stop();

    XINFO("[RESULT] client1 throughput %.2f reqs/s, latency avg %.2f us, "
          "min %.2f p50 %.2f us, p90 %.2f us, p99 %.2f us",
          (double)latency1.Cnt() * 1e9 / test_time,
          (double)latency1.Avg() / 1000,
          (double)latency1.Percentile(0.0) / 1000,
          (double)latency1.Percentile(0.5) / 1000,
          (double)latency1.Percentile(0.9) / 1000,
          (double)latency1.Percentile(0.99) / 1000);

    XINFO("[RESULT] client2 throughput %.2f reqs/s, latency avg %.2f us, "
          "min %.2f p50 %.2f us, p90 %.2f us, p99 %.2f us",
          (double)latency2.Cnt() * 1e9 / test_time,
          (double)latency2.Avg() / 1000,
          (double)latency2.Percentile(0.0) / 1000,
          (double)latency2.Percentile(0.5) / 1000,
          (double)latency2.Percentile(0.9) / 1000,
          (double)latency2.Percentile(0.99) / 1000);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        XINFO("Usage: %s <engine dir>", argv[0]);
        XERRO("lack arguments, abort...");
    }

    cudaStream_t stream1;
    cudaStream_t stream2;
	CudlaModel model1(argv[1]);
	CudlaModel model2(argv[1]);
    CUDART_ASSERT(cudaStreamCreate(&stream1));
    CUDART_ASSERT(cudaStreamCreate(&stream2));

    uint64_t xq1 = CudlaXQueueCreate(stream1, PREEMPT_MODE_STOP_SUBMISSION,
                                     1, 1);
    uint64_t xq2 = CudlaXQueueCreate(stream2, PREEMPT_MODE_STOP_SUBMISSION,
                                     1, 1);

    WarmUp(model1, stream1);
    WarmUp(model2, stream2);

    CudlaXQueueSetBandwidth(stream1, 3);
    CudlaXQueueSetBandwidth(stream2, 1);

    CoRun(model1, stream1, model2, stream2);

    CudlaXQueueDestroy(xq1);
    CudlaXQueueDestroy(xq2);
}
