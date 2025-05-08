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

static void RtTask(AclModel &model, aclrtStream stream)
{
    srand(time(NULL));

    ProcessSync psync;
    DataProcessor<int64_t> latency;

    WarmUp(model, stream);

    psync.Sync(2, "rt");

    int64_t ns = EXEC_TIME(nanoseconds, {
        for (int64_t i = 0; i < LATENCY_TEST_CNT; ++i) {
            auto next = std::chrono::system_clock::now() +
                        std::chrono::milliseconds(10 + rand() % 20);
            latency.Add(EXEC_TIME(nanoseconds, {
                model.Execute(stream);
            }));
            std::this_thread::sleep_until(next);
        }
    });

    psync.Sync(3, "rt");

    XINFO("[RESULT] rt throughput %.2f reqs/s, latency avg %.2f us, "
          "min %.2f p50 %.2f us, p90 %.2f us, p99 %.2f us",
          (double)LATENCY_TEST_CNT * 1e9 / ns,
          (double)latency.Avg() / 1000,
          (double)latency.Percentile(0.0) / 1000,
          (double)latency.Percentile(0.5) / 1000,
          (double)latency.Percentile(0.9) / 1000,
          (double)latency.Percentile(0.99) / 1000);
    
    latency.SaveCDF("hpf.base.cdf");
}

static void BeTask(AclModel &model, aclrtStream stream)
{
    ProcessSync psync;

    WarmUp(model, stream);

    psync.Sync(2, "be");

    int64_t infer_cnt = 0;
    int64_t ns = EXEC_TIME(nanoseconds, {
        while (psync.GetCnt() < 3) {
            model.Execute(stream);
            infer_cnt++;
        }
    });

    XINFO("[RESULT] be throughput %.2f reqs/s",
          (double)infer_cnt * 1e9 / ns);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        XINFO("Usage: %s <model dir>", argv[0]);
        XERRO("lack arguments, abort...");
    }

    // parent is rt task
    pid_t cpid = fork();

    aclrtStream stream;
    AclModel::InitResource();
    ACL_ASSERT(aclrtCreateStream(&stream));

    AclModel model(std::string(argv[1]) + "/slices");

    if (cpid) {
        // parent rt task
        RtTask(model, stream);
    } else {
        // child be task
        BeTask(model, stream);
    }
}
