#pragma once

#include "model.h"

#define LATENCY_TEST_CNT       1000
#define CORRECTNESS_TEST_CNT   500
#define SLEEP_US               800

void WarmUp(TRTModel &model, cudaStream_t stream);
int64_t InferLatencyNs(TRTModel &model, cudaStream_t stream);
void PreemptTest(TRTModel &model, cudaStream_t stream);

void CheckCorrectness(TRTModel &model, int32_t testcase);
void CorrectnessTest1(TRTModel &model, cudaStream_t stream);
void CorrectnessTest2(TRTModel &model, cudaStream_t stream);
void CorrectnessTest3(TRTModel &model, cudaStream_t stream,
                      int64_t enqueue_ns);
void CorrectnessTest4(TRTModel &model, cudaStream_t stream,
                      int64_t enqueue_ns, int64_t execute_ns);
void CorrectnessTest5(TRTModel &model, cudaStream_t stream,
                      int64_t enqueue_ns, int64_t execute_ns);
void CorrectnessTest6(TRTModel &model, cudaStream_t stream,
                      int64_t enqueue_ns, int64_t execute_ns);
void CorrectnessTest7(TRTModel &model, cudaStream_t stream,
                      int64_t enqueue_ns, int64_t execute_ns);
void CorrectnessTest(TRTModel &model, cudaStream_t stream);
