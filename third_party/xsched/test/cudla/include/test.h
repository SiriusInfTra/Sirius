#pragma once

#include "model.h"

#define LATENCY_TEST_CNT    500
#define SLEEP_US            800

void WarmUp(CudlaModel &model, cudaStream_t stream);
int64_t InferLatencyNs(CudlaModel &model, cudaStream_t stream);
void PreemptTest(CudlaModel &model, cudaStream_t stream);
