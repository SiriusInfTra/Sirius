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

int main(int argc, char **argv)
{
    if (argc < 2) {
        XINFO("Usage: %s <model dir>", argv[0]);
        XERRO("lack arguments, abort...");
    }

    pid_t cpid = fork();

    aclrtStream stream;
    AclModel::InitResource();
    ACL_ASSERT(aclrtCreateStream(&stream));

    AclModel model(std::string(argv[1]) + "/slices");

    ProcessSync psync;
    DataProcessor<int64_t> latency;

    WarmUp(model, stream);

    psync.Sync(2, cpid ? "client1" : "client2");

    int64_t ns = EXEC_TIME(nanoseconds, {
        for (int64_t i = 0; i < LATENCY_TEST_CNT; ++i) {
            latency.Add(EXEC_TIME(nanoseconds, {
                model.Execute(stream);
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
