#include "sde.h"
#include "test.h"
#include "utils.h"
#include "utils/log.h"
#include "utils/data.h"
#include "utils/timing.h"
#include "shim/vpi/xctrl.h"

#include <thread>
#include <vpi/Stream.h>

using namespace xsched::utils;

int main(int argc, char **argv)
{
    if (argc < 3) {
        XINFO("Usage: %s <left image> <right image>",
              argv[0]);
        XERRO("lack arguments, abort...");
    }

    DataProcessor<int64_t> latency;
    OfaSdeRunner runner(argv[1], argv[2]);
    VPIStream stream = runner.CreateStream();
    runner.Init(stream);

    WarmUp(&runner, stream, 30);

    for (int64_t i = 0; i < LATENCY_TEST_CNT; ++i) {
        auto next = std::chrono::system_clock::now() +
            std::chrono::milliseconds(40 + rand() % 20);
        latency.Add(EXEC_TIME(nanoseconds, {
            runner.Execute(stream, 8, false);
        }));
        std::this_thread::sleep_until(next);
    }

    int64_t ns = EXEC_TIME(nanoseconds, {
        for (int64_t i = 0; i < LATENCY_TEST_CNT; ++i) {
            runner.Execute(stream, 8, false);
        }
    });

    XINFO("[RESULT] max throughput %.2f reqs/s, latency avg %.2f us, "
          "min %.2f p50 %.2f us, p90 %.2f us, p99 %.2f us",
          (double)LATENCY_TEST_CNT * 1e9 / ns,
          (double)latency.Avg() / 1000,
          (double)latency.Percentile(0.0) / 1000,
          (double)latency.Percentile(0.5) / 1000,
          (double)latency.Percentile(0.9) / 1000,
          (double)latency.Percentile(0.99) / 1000);
    
    latency.SaveCDF("pva.standalone.cdf");
}
