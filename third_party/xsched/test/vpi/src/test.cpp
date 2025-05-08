#include "test.h"
#include "utils.h"
#include "utils/log.h"
#include "utils/data.h"
#include "utils/psync.h"
#include "utils/runner.h"
#include "utils/timing.h"
#include "shim/vpi/xctrl.h"

using namespace xsched::utils;

void WarmUp(VpiRunner *runner, VPIStream stream, const size_t qlen)
{
    for (size_t i = 0; i < LATENCY_TEST_CNT; ++i)
        runner->Execute(stream, qlen, false);
}

int64_t ExecuteLatencyNs(VpiRunner *runner,
                         VPIStream stream,
                         const size_t qlen,
                         const bool sync)
{
    int64_t exec_time = EXEC_TIME(nanoseconds, {
        for (int i = 0; i < LATENCY_TEST_CNT; ++i) {
            runner->Execute(stream, qlen, sync);
        }
    }) / LATENCY_TEST_CNT;
    return exec_time;
}

void PreemptTest(VpiRunner *runner, VPIStream stream, const size_t qlen)
{
    WarmUp(runner, stream, qlen);

    srand(time(0));

    LoopRunner loop;
    DataProcessor<int64_t> preempt_latency;
    DataProcessor<int64_t> restore_latency;

    loop.Start([&]() -> void {
        // enqueue compute tasks to gpu and make it busy
        runner->Execute(stream, qlen, false);
    });

    std::this_thread::sleep_for(std::chrono::seconds(2));
    for (size_t i = 0; i < 3 * LATENCY_TEST_CNT; ++i) {
        int64_t sleep_us = rand() % SLEEP_US;
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
        int64_t preempt_ns = EXEC_TIME(nanoseconds, {
            VpiXQueueSuspend(stream, true);
        });
        sleep_us = (rand() % SLEEP_US) + SLEEP_US;
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
        int64_t restore_ns = EXEC_TIME(nanoseconds, {
            VpiXQueueResume(stream);
        });

        // warmup and cooldown
        if (i < LATENCY_TEST_CNT && i >= 2 * LATENCY_TEST_CNT) continue;
        preempt_latency.Add(preempt_ns);
        restore_latency.Add(restore_ns);
    }

    loop.Stop();

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

void CoRun(VpiRunner *runner_rt, VPIStream stream_rt, const size_t qlen_rt,
           VpiRunner *runner_be, VPIStream stream_be, const size_t qlen_be)
{
    srand(time(NULL));

    LoopRunner loop;
    loop.Start([&]() -> void {
        runner_be->Execute(stream_be, qlen_be, false);
    });

    DataProcessor<int64_t> latency;
    std::this_thread::sleep_for(std::chrono::seconds(2));

    for (int i = 0; i < LATENCY_TEST_CNT; ++i) {
        latency.Add(EXEC_TIME(nanoseconds, {
            runner_rt->Execute(stream_rt, qlen_rt, false);
        }));
        std::this_thread::sleep_for(std::chrono::microseconds(
            SLEEP_US / 2 + rand() % SLEEP_US));
    }
    loop.Stop();

    XINFO("[RESULT] latency avg %.2f us, p50 %.2f us, p90 %.2f us, p99 %.2f us",
          (double)latency.Avg() / 1000,
          (double)latency.Percentile(0.5) / 1000,
          (double)latency.Percentile(0.9) / 1000,
          (double)latency.Percentile(0.99) / 1000);
}

void CbsCoRun(VpiRunner *runner1, VPIStream stream1, const size_t qlen1,
              VpiRunner *runner2, VPIStream stream2, const size_t qlen2)
{
    DataProcessor<int64_t> latency1;
    DataProcessor<int64_t> latency2;

    bool begin = false;

    LoopRunner runner;
    runner.Start([&]() -> void {
        int64_t exec_time = EXEC_TIME(nanoseconds, {
            runner2->Execute(stream2, qlen2, false);
        });
        if (begin) latency2.Add(exec_time);
    });

    std::this_thread::sleep_for(std::chrono::seconds(2));

    begin = true;
    double test_time = EXEC_TIME(nanoseconds, {
        for (int i = 0; i < LATENCY_TEST_CNT; ++i) {
            latency1.Add(EXEC_TIME(nanoseconds, {
                runner1->Execute(stream1, qlen1, false);
            }));
        }
    });
    runner.Stop();

    XINFO("[RESULT] client1 throughput %.2f reqs/s, latency avg %.2f us, "
          "min %.2f p50 %.2f us, p90 %.2f us, p99 %.2f us",
          (double)latency1.Cnt() * 1e9 / test_time,
          (double)latency1.Avg() / 1000,
          (double)latency1.Percentile(0.0) / 1000,
          (double)latency1.Percentile(0.5) / 1000,
          (double)latency1.Percentile(0.9) / 1000,
          (double)latency1.Percentile(0.99) / 1000);

    XINFO("[RESULT] client2 throughput %.2f reqs/s, latency avg %.2f us, "
          "min %.2f p50 %.2f us, p90 %.2f us, p99 %.2f us",
          (double)latency2.Cnt() * 1e9 / test_time,
          (double)latency2.Avg() / 1000,
          (double)latency2.Percentile(0.0) / 1000,
          (double)latency2.Percentile(0.5) / 1000,
          (double)latency2.Percentile(0.9) / 1000,
          (double)latency2.Percentile(0.99) / 1000);
}

void CbsTask(VpiRunner *runner, VPIStream stream,
             const size_t qlen, const char *name)
{
    ProcessSync psync;
    DataProcessor<int64_t> latency;

    WarmUp(runner, stream, qlen);

    psync.Sync(2, name);

    int64_t ns = EXEC_TIME(nanoseconds, {
        for (int64_t i = 0; i < LATENCY_TEST_CNT; ++i) {
            latency.Add(EXEC_TIME(nanoseconds, {
                runner->Execute(stream, qlen, false);
            }));
        }
    });

    XINFO("[RESULT] %s throughput %.2f reqs/s, latency avg %.2f us, "
          "min %.2f p50 %.2f us, p90 %.2f us, p99 %.2f us",
          name,
          (double)LATENCY_TEST_CNT * 1e9 / ns,
          (double)latency.Avg() / 1000,
          (double)latency.Percentile(0.0) / 1000,
          (double)latency.Percentile(0.5) / 1000,
          (double)latency.Percentile(0.9) / 1000,
          (double)latency.Percentile(0.99) / 1000);
}

void HpfCoRun(VpiRunner *runner_rt, VPIStream stream_rt, const size_t qlen_rt,
              VpiRunner *runner_be, VPIStream stream_be, const size_t qlen_be,
              int64_t base_wait, int64_t rand_wait)
{
    srand(time(NULL));

    bool begin = false;
    int64_t be_cnt = 0;

    LoopRunner runner;
    runner.Start([&]() -> void {
        if (begin) be_cnt++;
        runner_be->Execute(stream_be, qlen_be, false);
    });

    DataProcessor<int64_t> latency;
    std::this_thread::sleep_for(std::chrono::seconds(2));

    begin = true;
    double test_time = EXEC_TIME(nanoseconds, {
        for (int i = 0; i < LATENCY_TEST_CNT; ++i) {
            auto next = std::chrono::system_clock::now() +
                std::chrono::milliseconds(base_wait + rand() % rand_wait);
            latency.Add(EXEC_TIME(nanoseconds, {
                runner_rt->Execute(stream_rt, qlen_rt, false);
            }));
            std::this_thread::sleep_until(next);
        }
    });
    runner.Stop();

    XINFO("[RESULT] be throughput %.2f reqs/s",
          (double)be_cnt * 1e9 / test_time);
    XINFO("[RESULT] rt throughput %.2f reqs/s, latency avg %.2f us, "
          "min %.2f p50 %.2f us, p90 %.2f us, p99 %.2f us",
          (double)LATENCY_TEST_CNT * 1e9 / test_time,
          (double)latency.Avg() / 1000,
          (double)latency.Percentile(0.0) / 1000,
          (double)latency.Percentile(0.5) / 1000,
          (double)latency.Percentile(0.9) / 1000,
          (double)latency.Percentile(0.99) / 1000);
    
    latency.SaveCDF("hpf.sched.cdf");
}

void RtTask(VpiRunner *runner, VPIStream stream, const size_t qlen,
            int64_t base_wait, int64_t rand_wait)
{
    srand(time(NULL));

    ProcessSync psync;
    DataProcessor<int64_t> latency;

    WarmUp(runner, stream, qlen);

    psync.Sync(2, "rt");

    int64_t ns = EXEC_TIME(nanoseconds, {
        for (int64_t i = 0; i < LATENCY_TEST_CNT; ++i) {
            auto next = std::chrono::system_clock::now() +
                std::chrono::milliseconds(base_wait + rand() % rand_wait);
            latency.Add(EXEC_TIME(nanoseconds, {
                runner->Execute(stream, qlen, false);
            }));
            std::this_thread::sleep_until(next);
        }
    });

    psync.Sync(3, "rt");

    XINFO("[RESULT] rt throughput %.2f reqs/s, latency avg %.2f us, "
          "min %.2f p50 %.2f us, p90 %.2f us, p99 %.2f us",
          (double)LATENCY_TEST_CNT * 1e9 / ns,
          (double)latency.Avg() / 1000,
          (double)latency.Percentile(0.0) / 1000,
          (double)latency.Percentile(0.5) / 1000,
          (double)latency.Percentile(0.9) / 1000,
          (double)latency.Percentile(0.99) / 1000);
    
    latency.SaveCDF("hpf.base.cdf");
}

void BeTask(VpiRunner *runner, VPIStream stream, const size_t qlen)
{
    ProcessSync psync;

    WarmUp(runner, stream, qlen);

    psync.Sync(2, "be");

    int64_t infer_cnt = 0;
    int64_t ns = EXEC_TIME(nanoseconds, {
        while (psync.GetCnt() < 3) {
            runner->Execute(stream, qlen, false);
            infer_cnt++;
        }
    });

    XINFO("[RESULT] be throughput %.2f reqs/s",
          (double)infer_cnt * 1e9 / ns);
}
