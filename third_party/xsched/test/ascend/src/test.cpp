#include "test.h"
#include "utils/log.h"
#include "utils/data.h"
#include "utils/runner.h"
#include "utils/timing.h"
#include "shim/ascend/xctrl.h"

using namespace xsched::utils;

void WarmUp(AclModel &model, aclrtStream stream)
{
    for (size_t i = 0; i < LATENCY_TEST_CNT; ++i) model.Execute(stream);
}

int64_t InferLatencyNs(AclModel &model, aclrtStream stream)
{
    int64_t exec_time = EXEC_TIME(nanoseconds, {
        for (int i = 0; i < LATENCY_TEST_CNT; ++i) {
            model.Execute(stream);
        }
    }) / LATENCY_TEST_CNT;
    return exec_time;
}

void PreemptTest(AclModel &model, aclrtStream stream)
{
    WarmUp(model, stream);

    srand(time(0));

    LoopRunner runner;
    DataProcessor<int64_t> preempt_latency;
    DataProcessor<int64_t> restore_latency;

    runner.Start([&]() -> void {
        // enqueue compute tasks to gpu and make it busy
        model.Execute(stream);
    });

    std::this_thread::sleep_for(std::chrono::seconds(2));
    for (size_t i = 0; i < 3 * LATENCY_TEST_CNT; ++i) {
        int64_t sleep_us = rand() % SLEEP_US;
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
        int64_t preempt_ns = EXEC_TIME(nanoseconds, {
            AclXQueueSuspend(stream, true);
        });
        sleep_us = (rand() % SLEEP_US) + SLEEP_US;
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
        int64_t restore_ns = EXEC_TIME(nanoseconds, {
            AclXQueueResume(stream);
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
