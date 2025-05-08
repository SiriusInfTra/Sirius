#include "test.h"
#include "utils.h"
#include "model.h"
#include "utils/log.h"
#include "shim/ascend/xctrl.h"

#include <acl/acl.h>

int main(int argc, char **argv)
{
	if (argc < 2) {
        XINFO("Usage: %s <model dir>", argv[0]);
        XERRO("lack arguments, abort...");
	}

    aclrtStream stream;
    AclModel::InitResource();
    ACL_ASSERT(aclrtCreateStream(&stream));

    XINFO("[TEST] inference baseline test");
    {
        AclModel model(argv[1]);
        WarmUp(model, stream);
        int64_t origin_ns = InferLatencyNs(model, stream);
        XINFO("[RESULT] original model: %ldus", origin_ns / 1000);
    }

    AclModel model(std::string(argv[1]) + "/slices");
    WarmUp(model, stream);
    int64_t disable_ns = InferLatencyNs(model, stream);
    XINFO("[RESULT] disable preemption: %ldus", disable_ns / 1000);

    XINFO("[TEST] inference overhead test");
    XINFO("[TEST] AclXQueueCreate()");
    uint64_t handle = AclXQueueCreate(stream, PREEMPT_MODE_STOP_SUBMISSION,
                                      4, 2);

    WarmUp(model, stream);
    int64_t enable_ns = InferLatencyNs(model, stream);
    XINFO("[RESULT] enable preemption: %ldus", enable_ns / 1000);
    XINFO("[RESULT] overhead: %ldus, %.2f%%",
          (enable_ns - disable_ns) / 1000,
          double(enable_ns - disable_ns) / double(disable_ns) * 100);

    XINFO("[TEST] inference preempt test");
    PreemptTest(model, stream);

    XINFO("[TEST] test complete");
    XINFO("[TEST] AclXQueueDestroy()");
    AclXQueueDestroy(handle);
    ACL_ASSERT(aclrtDestroyStream(stream));
    return 0;
}
