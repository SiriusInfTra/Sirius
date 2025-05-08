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

static void CoRun(AclModel &model_rt, aclrtStream stream_rt,
                  AclModel &model_be, aclrtStream stream_be)
{
    srand(time(NULL));

    bool begin = false;
    int64_t be_cnt = 0;

    LoopRunner runner;
    runner.Start([&]() -> void {
        if (begin) be_cnt++;
        model_be.Execute(stream_be);
    });

    DataProcessor<int64_t> latency;
    std::this_thread::sleep_for(std::chrono::seconds(2));

    begin = true;
    double test_time = EXEC_TIME(nanoseconds, {
        for (int i = 0; i < LATENCY_TEST_CNT; ++i) {
            auto next = std::chrono::system_clock::now() +
                        std::chrono::milliseconds(10 + rand() % 20);
            latency.Add(EXEC_TIME(nanoseconds, {
                model_rt.Execute(stream_rt);
            }));
            std::this_thread::sleep_until(next);
        }
    });
    runner.Stop();

    XINFO("[RESULT] be throughput %.2f reqs/s",
          (double)be_cnt * 1e9 / test_time);
    XINFO("[RESULT] rt throughput %.2f reqs/s, latency avg %.2f us, "
          "min %.2f p50 %.2f us, p90 %.2f us, p99 %.2f us",
          (double)LATENCY_TEST_CNT * 1e9 / test_time,
          (double)latency.Avg() / 1000,
          (double)latency.Percentile(0.0) / 1000,
          (double)latency.Percentile(0.5) / 1000,
          (double)latency.Percentile(0.9) / 1000,
          (double)latency.Percentile(0.99) / 1000);
    
    latency.SaveCDF("hpf.sched.cdf");
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        XINFO("Usage: %s <model dir>", argv[0]);
        XERRO("lack arguments, abort...");
    }

    aclrtStream stream_rt;
    aclrtStream stream_be;
    AclModel::InitResource();
    ACL_ASSERT(aclrtCreateStream(&stream_rt));
    ACL_ASSERT(aclrtCreateStream(&stream_be));

    AclModel model_rt(std::string(argv[1]) + "/slices");
    AclModel model_be(std::string(argv[1]) + "/slices");

    uint64_t xq_rt = AclXQueueCreate(stream_rt, PREEMPT_MODE_STOP_SUBMISSION,
                                     64, 32);
    uint64_t xq_be = AclXQueueCreate(stream_be, PREEMPT_MODE_STOP_SUBMISSION,
                                     3, 3);

    WarmUp(model_rt, stream_rt);
    WarmUp(model_be, stream_be);

    AclXQueueSetPriority(stream_rt, 2);
    AclXQueueSetPriority(stream_be, 1);

    CoRun(model_rt, stream_rt, model_be, stream_be);

    AclXQueueDestroy(xq_rt);
    AclXQueueDestroy(xq_be);
}
