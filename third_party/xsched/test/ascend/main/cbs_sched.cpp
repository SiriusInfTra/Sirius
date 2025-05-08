#include "test.h"
#include "utils.h"
#include "model.h"
#include "utils/log.h"
#include "utils/data.h"
#include "utils/psync.h"
#include "utils/timing.h"
#include "utils/runner.h"
#include "shim/ascend/xctrl.h"

#include <thread>
#include <acl/acl.h>

using namespace xsched::utils;

static void CoRun(AclModel &model1, aclrtStream stream1,
                  AclModel &model2, aclrtStream stream2)
{
    DataProcessor<int64_t> latency1;
    DataProcessor<int64_t> latency2;

    bool begin = false;

    LoopRunner runner;
    runner.Start([&]() -> void {
        int64_t exec_time = EXEC_TIME(nanoseconds, {
            model2.Execute(stream2);
        });
        if (begin) latency2.Add(exec_time);
    });

    std::this_thread::sleep_for(std::chrono::seconds(2));

    begin = true;
    double test_time = EXEC_TIME(nanoseconds, {
        for (int i = 0; i < LATENCY_TEST_CNT; ++i) {
            latency1.Add(EXEC_TIME(nanoseconds, {
                model1.Execute(stream1);
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
        XINFO("Usage: %s <model dir>", argv[0]);
        XERRO("lack arguments, abort...");
    }

    aclrtStream stream1;
    aclrtStream stream2;
    AclModel::InitResource();
    ACL_ASSERT(aclrtCreateStream(&stream1));
    ACL_ASSERT(aclrtCreateStream(&stream2));

    AclModel model1(std::string(argv[1]) + "/slices");
    AclModel model2(std::string(argv[1]) + "/slices");

    uint64_t xq1 = AclXQueueCreate(stream1, PREEMPT_MODE_STOP_SUBMISSION,
                                   4, 3);
    uint64_t xq2 = AclXQueueCreate(stream2, PREEMPT_MODE_STOP_SUBMISSION,
                                   4, 3);

    WarmUp(model1, stream1);
    WarmUp(model2, stream2);

    AclXQueueSetBandwidth(stream1, 3);
    AclXQueueSetBandwidth(stream2, 1);

    CoRun(model1, stream1, model2, stream2);

    AclXQueueDestroy(xq1);
    AclXQueueDestroy(xq2);
}
