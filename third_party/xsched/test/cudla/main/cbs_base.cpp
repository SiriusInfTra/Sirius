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

int main(int argc, char **argv)
{
    if (argc < 2) {
        XINFO("Usage: %s <engine dir>", argv[0]);
        XERRO("lack arguments, abort...");
    }

    // parent is rt task
    pid_t cpid = fork();

	CudlaModel model(argv[1]);
    cudaStream_t stream;
    CUDART_ASSERT(cudaStreamCreate(&stream));

    ProcessSync psync;
    DataProcessor<int64_t> latency;

    WarmUp(model, stream);

    psync.Sync(2, cpid ? "client1" : "client2");

    int64_t ns = EXEC_TIME(nanoseconds, {
        for (int64_t i = 0; i < LATENCY_TEST_CNT; ++i) {
            latency.Add(EXEC_TIME(nanoseconds, {
                model.Infer(stream);
            }));
        }
    });

    XINFO("[RESULT] %s throughput %.2f reqs/s, latency avg %.2f us, "
          "min %.2f p50 %.2f us, p90 %.2f us, p99 %.2f us",
          cpid ? "client1" : "client2",
          (double)LATENCY_TEST_CNT * 1e9 / ns,
          (double)latency.Avg() / 1000,
          (double)latency.Percentile(0.0) / 1000,
          (double)latency.Percentile(0.5) / 1000,
          (double)latency.Percentile(0.9) / 1000,
          (double)latency.Percentile(0.99) / 1000);
}
