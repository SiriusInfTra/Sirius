#include "sched/protocol/protocol.h"

using namespace xsched::sched;
using namespace xsched::preempt;

const std::string xsched::sched::kPolicyEnvVarName = "XSCHED_POLICY";
const std::string xsched::sched::kServerChannelName = "xsched-server";
const std::string xsched::sched::kClientChannelPrefix = "xsched-client-";

/// A new policy name should be added here
/// when creating a new policy.
const std::map<std::string, PolicyType> xsched::sched::kPolicyTypeMap {
    { "global"                    , kPolicyGlobal                  },
    { "user_managed"              , kPolicyUserManaged             },
    { "highest_priority_first"    , kPolicyHighestPriorityFirst    },
    { "constant_bandwidth_server" , kPolicyConstantBandwidthServer },
};

const std::map<XDevice, std::string> xsched::sched::kDeviceMap {
    { XDevice::kDeviceVPI    , "VPI"    },
    { XDevice::kDeviceCUDA   , "CUDA"   },
    { XDevice::kDeviceCUDLA  , "cuDLA"  },
    { XDevice::kDeviceAscend , "Ascend" },
};

const std::map<std::string, RequestType> xsched::sched::kRequestTypeMap {
    { "query_xqueue_status" , kRequestTypeQueryXQueueStatus },
    { "give_hint"           , kRequestTypeGiveHint          },
};
