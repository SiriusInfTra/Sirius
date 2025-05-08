#include "test.h"
#include "utils.h"
#include "model.h"
#include "utils/log.h"
#include "utils/data.h"
#include "utils/timing.h"
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

    aclrtStream stream;
    AclModel::InitResource();
    ACL_ASSERT(aclrtCreateStream(&stream));

    DataProcessor<int64_t> latency;
    AclModel model(std::string(argv[1]) + "/slices");

    WarmUp(model, stream);

    for (int64_t i = 0; i < LATENCY_TEST_CNT; ++i) {
        latency.Add(EXEC_TIME(nanoseconds, {
            model.Execute(stream);
        }));
    }

    int64_t ns = EXEC_TIME(nanoseconds, {
        for (int64_t i = 0; i < LATENCY_TEST_CNT; ++i) {
            model.Enqueue(stream);
        }
        ACL_ASSERT(aclrtSynchronizeStream(stream));
    });

    XINFO("[RESULT] max throughput %.2f reqs/s, latency avg %.2f us, "
          "min %.2f p50 %.2f us, p90 %.2f us, p99 %.2f us",
          (double)LATENCY_TEST_CNT * 1e9 / ns,
          (double)latency.Avg() / 1000,
          (double)latency.Percentile(0.0) / 1000,
          (double)latency.Percentile(0.5) / 1000,
          (double)latency.Percentile(0.9) / 1000,
          (double)latency.Percentile(0.99) / 1000);
    
    latency.SaveCDF("ascend.standalone.cdf");
}
