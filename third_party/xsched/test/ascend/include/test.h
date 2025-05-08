#pragma once

#include "model.h"

#define LATENCY_TEST_CNT    1000
#define SLEEP_US            800

void WarmUp(AclModel &model, aclrtStream stream);
int64_t InferLatencyNs(AclModel &model, aclrtStream stream);
void PreemptTest(AclModel &model, aclrtStream stream);
