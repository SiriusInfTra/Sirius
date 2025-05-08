#pragma once

#include "runner.h"

#define LATENCY_TEST_CNT    500
#define SLEEP_US            800

void WarmUp(VpiRunner *runner, VPIStream stream, const size_t qlen);
int64_t ExecuteLatencyNs(VpiRunner *runner,
                         VPIStream stream,
                         const size_t qlen,
                         const bool sync);
void PreemptTest(VpiRunner *runner, VPIStream stream, const size_t qlen);
void CoRun(VpiRunner *runner_rt, VPIStream stream_rt, const size_t qlen_rt,
           VpiRunner *runner_be, VPIStream stream_be, const size_t qlen_be);

void CbsCoRun(VpiRunner *runner1, VPIStream stream1, const size_t qlen1,
              VpiRunner *runner2, VPIStream stream2, const size_t qlen2);
void CbsTask(VpiRunner *runner, VPIStream stream,
             const size_t qlen, const char *name);
void HpfCoRun(VpiRunner *runner_rt, VPIStream stream_rt, const size_t qlen_rt,
              VpiRunner *runner_be, VPIStream stream_be, const size_t qlen_be,
              int64_t base_wait, int64_t rand_wait);
void RtTask(VpiRunner *runner, VPIStream stream, const size_t qlen,
            int64_t base_wait, int64_t rand_wait);
void BeTask(VpiRunner *runner, VPIStream stream, const size_t qlen);
