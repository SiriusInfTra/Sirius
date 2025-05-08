#include "test.h"
#include "utils/log.h"
#include "utils/data.h"
#include "utils/runner.h"
#include "utils/timing.h"
#include "shim/cuda/xctrl.h"

using namespace xsched::utils;

void WarmUp(TRTModel &model, cudaStream_t stream)
{
    for (size_t i = 0; i < LATENCY_TEST_CNT; ++i) model.Infer(stream);
}

int64_t InferLatencyNs(TRTModel &model, cudaStream_t stream)
{
    int64_t exec_time = EXEC_TIME(nanoseconds, {
        for (int i = 0; i < LATENCY_TEST_CNT; ++i) {
            // simulate real inference routine:
            //     copy input tensors to device,
            //     compute,
            //     copy output tensors to host.
            model.CopyInputAsync(stream);
            model.InferAsync(stream);
            model.CopyOutputAsync(stream);
            cudaStreamSynchronize(stream);
        }
    }) / LATENCY_TEST_CNT;
    return exec_time;
}

void PreemptTest(TRTModel &model, cudaStream_t stream)
{
    WarmUp(model, stream);

    srand(time(0));

    LoopRunner runner;
    DataProcessor<int64_t> preempt_latency;
    DataProcessor<int64_t> restore_latency;

    runner.Start([&]() -> void {
        // enqueue compute tasks to gpu and make it busy
        model.Infer(stream);
    });

    std::this_thread::sleep_for(std::chrono::seconds(2));
    for (size_t i = 0; i < 3 * LATENCY_TEST_CNT; ++i) {
        int64_t sleep_us = rand() % SLEEP_US;
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
        int64_t preempt_ns = EXEC_TIME(nanoseconds, {
            CudaXQueuePreempt(stream, true);
        });
        sleep_us = (rand() % SLEEP_US) + SLEEP_US;
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
        int64_t restore_ns = EXEC_TIME(nanoseconds, {
            CudaXQueueResume(stream);
        });

        // warmup and cooldown
        if (i < LATENCY_TEST_CNT && i >= 2 * LATENCY_TEST_CNT) continue;
        preempt_latency.Add(preempt_ns);
        restore_latency.Add(restore_ns);
    }

    runner.Stop();

    double restore_avg = restore_latency.Avg() / 1000.0;
    double preempt_avg = preempt_latency.Avg() / 1000.0;
    double p50 = preempt_latency.Percentile(0.50) / 1000.0;
    double p95 = preempt_latency.Percentile(0.95) / 1000.0;
    double p99 = preempt_latency.Percentile(0.99) / 1000.0;

    XINFO("[RESULT] restore avg: %.2f", restore_avg);
    XINFO("[RESULT] preempt avg: %.2f", preempt_avg);
    XINFO("[RESULT] preempt p50: %.2f", p50);
    XINFO("[RESULT] preempt p95: %.2f", p95);
    XINFO("[RESULT] preempt p99: %.2f", p99);
}

void CheckCorrectness(TRTModel &model, int32_t testcase)
{
    if (model.CheckOutput()) {
        XINFO("[RESULT] [PASS] testcase %d PASSED", testcase);
    } else {
        XWARN("[RESULT] [FAIL] testcase %d FAILED", testcase);
    }
}

void CorrectnessTest1(TRTModel &model, cudaStream_t stream)
{
    // TEST CASE 1
    // -----------|<------ enqueue + execute ------>|<------------------ execute ------------------>|-----------
    model.ClearOutput(stream);
    model.Infer(stream);
    model.CopyOutput(stream);
    CheckCorrectness(model, 1);
}

void CorrectnessTest2(TRTModel &model, cudaStream_t stream)
{
    // TEST CASE 2
    // preempt
    //    |  restore
    //    |     |
    //    v     v
    // -----------|<------ enqueue + execute ------>|<------------------ execute ------------------>|-----------
    CudaXQueuePreempt(stream, true);
    CudaXQueueResume(stream);

    model.ClearOutput(stream);
    model.Infer(stream);
    model.CopyOutput(stream);
    CheckCorrectness(model, 2);
}

void CorrectnessTest3(TRTModel &model, cudaStream_t stream, int64_t enqueue_ns)
{
    // TEST CASE 3
    // preempt                 restore
    //    |                       |
    //    v                       v
    // -----------|<-- enqueue -->|<-- enqueue + execute -->|<------------------ execute ------------------>|-----------
    bool pass = true;
    for (size_t i = 0; i < CORRECTNESS_TEST_CNT; ++i) {
        CudaXQueuePreempt(stream, true);
        int64_t sleep_ns = rand() % enqueue_ns;
        auto start = std::chrono::system_clock::now();
        std::thread thread([&]() -> void {
            std::this_thread::sleep_until(
                start + std::chrono::nanoseconds(sleep_ns));
            CudaXQueueResume(stream);
        });

        model.ClearOutput(stream);
        model.Infer(stream);
        model.CopyOutput(stream);
        thread.join();

        if (!model.CheckOutput()) {
            pass = false;
            XWARN("[RESULT] [FAIL] testcase 3 FAILED, "
                  "restore after preempt time: %ldus", sleep_ns / 1000);
            break;
        }
    }
    if (pass) {
        XINFO("[RESULT] [PASS] testcase 3 PASSED");
    }
}

void CorrectnessTest4(TRTModel &model, cudaStream_t stream,
                      int64_t enqueue_ns, int64_t execute_ns)
{
    // TEST CASE 4
    // preempt                                      restore
    //    |                                            |
    //    v                                            v
    // -----------|<------ enqueue ------>|<-- idle -->|<------------- execute ------------->|-----------
    bool pass = true;
    for (size_t i = 0; i < CORRECTNESS_TEST_CNT; ++i) {
        CudaXQueuePreempt(stream, true);
        int64_t sleep_ns = (rand() % (execute_ns - enqueue_ns)) + enqueue_ns;
        auto start = std::chrono::system_clock::now();
        std::thread thread([&]() -> void {
            std::this_thread::sleep_until(
                start + std::chrono::nanoseconds(sleep_ns));
            CudaXQueueResume(stream);
        });

        model.ClearOutput(stream);
        model.Infer(stream);
        model.CopyOutput(stream);
        thread.join();

        if (!model.CheckOutput()) {
            pass = false;
            XWARN("[RESULT] [FAIL] testcase 4 FAILED, "
                  "restore after preempt time: %ldus", sleep_ns / 1000);
            break;
        }
    }
    if (pass) {
        XINFO("[RESULT] [PASS] testcase 4 PASSED");
    }
}

void CorrectnessTest5(TRTModel &model, cudaStream_t stream,
                      int64_t enqueue_ns, int64_t execute_ns)
{
    // TEST CASE 5
    //                                   preempt         restore
    //                                      |               |
    //                                      v               v
    // -----------|<-- enqueue + execute -->|<-- enqueue -->|<-- enqueue + execute -->|<----------- execute ----------->|-----------
    bool pass = true;
    for (int i = 0; i < CORRECTNESS_TEST_CNT; ++i) {
        int64_t preempt_sleep_ns = rand() % enqueue_ns;
        int64_t restore_sleep_ns = (rand() % (enqueue_ns - preempt_sleep_ns))
                                 + preempt_sleep_ns;
        auto start = std::chrono::system_clock::now();
        std::thread thread([&]() -> void {
            std::this_thread::sleep_until(
                start + std::chrono::nanoseconds(preempt_sleep_ns));
            CudaXQueuePreempt(stream, true);
            std::this_thread::sleep_until(
                start + std::chrono::nanoseconds(restore_sleep_ns));
            CudaXQueueResume(stream);
        });

        model.ClearOutput(stream);
        model.Infer(stream);
        model.CopyOutput(stream);
        thread.join();

        if (!model.CheckOutput()) {
            pass = false;
            XWARN("[RESULT] [FAIL] testcase 5 FAILED, "
                  "preempt after enqueue time: %ldus, "
                  "restore after preempt time: %ldus",
                  preempt_sleep_ns / 1000, restore_sleep_ns / 1000);
            break;
        }
    }
    if (pass) {
        XINFO("[RESULT] [PASS] testcase 5 PASSED");
    }
}

void CorrectnessTest6(TRTModel &model, cudaStream_t stream,
                      int64_t enqueue_ns, int64_t execute_ns)
{
    // TEST CASE 6
    //                                   preempt                      restore
    //                                      |                            |
    //                                      v                            v
    // -----------|<-- enqueue + execute -->|<-- enqueue -->|<-- idle -->|<------------------ execute ------------------>|-----------
    bool pass = true;
    for (int i = 0; i < CORRECTNESS_TEST_CNT; ++i) {
        int64_t preempt_sleep_ns = rand() % enqueue_ns;
        int64_t restore_sleep_ns = (rand() % execute_ns) + enqueue_ns;
        auto start = std::chrono::system_clock::now();
        std::thread thread([&]() -> void {
            std::this_thread::sleep_until(
                start + std::chrono::nanoseconds(preempt_sleep_ns));
            CudaXQueuePreempt(stream, true);
            std::this_thread::sleep_until(
                start + std::chrono::nanoseconds(restore_sleep_ns));
            CudaXQueueResume(stream);
        });

        model.ClearOutput(stream);
        model.Infer(stream);
        model.CopyOutput(stream);
        thread.join();

        if (!model.CheckOutput()) {
            pass = false;
            XWARN("[RESULT] [FAIL] testcase 6 FAILED, "
                  "preempt after enqueue time: %ldus, "
                  "restore after preempt time: %ldus",
                  preempt_sleep_ns / 1000, restore_sleep_ns / 1000);
            break;
        }
    }
    if (pass) {
        XINFO("[RESULT] [PASS] testcase 6 PASSED");
    }
}

void CorrectnessTest7(TRTModel &model, cudaStream_t stream,
                      int64_t enqueue_ns, int64_t execute_ns)
{
    // TEST CASE 7
    //                                                   preempt      restore
    //                                                      |            |
    //                                                      v            v
    // -----------|<-- enqueue + execute -->|<-- execute -->|<-- idle -->|<------------------ execute ------------------>|-----------
    bool pass = true;
    for (int i = 0; i < CORRECTNESS_TEST_CNT; ++i) {
        int64_t preempt_sleep_ns = (rand() % (execute_ns - enqueue_ns))
                                 + enqueue_ns;
        int64_t restore_sleep_ns = (rand() % execute_ns) + preempt_sleep_ns;
        auto start = std::chrono::system_clock::now();
        std::thread thread([&]() -> void {
            std::this_thread::sleep_until(
                start + std::chrono::nanoseconds(preempt_sleep_ns));
            CudaXQueuePreempt(stream, true);
            std::this_thread::sleep_until(
                start + std::chrono::nanoseconds(restore_sleep_ns));
            CudaXQueueResume(stream);
        });

        model.ClearOutput(stream);
        model.Infer(stream);
        model.CopyOutput(stream);
        thread.join();

        if (!model.CheckOutput()) {
            pass = false;
            XWARN("[RESULT] [FAIL] testcase 7 FAILED, "
                  "preempt after enqueue time: %ldus, "
                  "restore after preempt time: %ldus",
                  preempt_sleep_ns / 1000, restore_sleep_ns / 1000);
            break;
        }
    }
    if (pass) {
        XINFO("[RESULT] [PASS] testcase 7 PASSED");
    }
}

void CorrectnessTest(TRTModel &model, cudaStream_t stream)
{
    srand(time(0));
    XINFO("[TEST] correctness test count: %d", CORRECTNESS_TEST_CNT);
    
    int64_t enqueue_ns = 0;
    int64_t execute_ns = EXEC_TIME(nanoseconds, {
        for (int i = 0; i < LATENCY_TEST_CNT; ++i) {
            // simulate real inference routine:
            //     copy input tensors to device,
            //     compute,
            //     copy output tensors to host.
            enqueue_ns += EXEC_TIME(nanoseconds, {
                model.CopyInputAsync(stream);
                model.InferAsync(stream);
                model.CopyOutputAsync(stream);
            });
            cudaStreamSynchronize(stream);
        }
    }) / LATENCY_TEST_CNT;
    enqueue_ns = enqueue_ns / LATENCY_TEST_CNT;

    CorrectnessTest1(model, stream);
    CorrectnessTest2(model, stream);
    CorrectnessTest3(model, stream, enqueue_ns);
    CorrectnessTest4(model, stream, enqueue_ns, execute_ns);
    CorrectnessTest5(model, stream, enqueue_ns, execute_ns);
    CorrectnessTest6(model, stream, enqueue_ns, execute_ns);
    CorrectnessTest7(model, stream, enqueue_ns, execute_ns);
}
