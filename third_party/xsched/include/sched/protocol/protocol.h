#pragma once

#include <map>
#include <string>

#include "sched/policy/policy.h"
#include "preempt/xqueue/xtype.h"

namespace xsched::sched
{

extern const std::string kPolicyEnvVarName;
extern const std::string kServerChannelName;
extern const std::string kClientChannelPrefix;
extern const std::map<std::string, PolicyType> kPolicyTypeMap;
extern const std::map<preempt::XDevice, std::string> kDeviceMap;

enum RequestType
{
    kRequestTypeQueryXQueueStatus = 0,
    kRequestTypeGiveHint          = 1,
};

extern const std::map<std::string, RequestType> kRequestTypeMap;

} // namespace xsched::sched
