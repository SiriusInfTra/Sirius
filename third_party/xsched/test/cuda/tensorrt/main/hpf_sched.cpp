#include "test.h"
#include "utils.h"
#include "model.h"
#include "utils/log.h"
#include "utils/data.h"
#include "utils/timing.h"
#include "utils/runner.h"
#include "shim/cuda/xctrl.h"

#include <thread>
#include <cuda_runtime.h>

using namespace xsched::utils;

static void CoRun(TRTModel &model_rt, cudaStream_t stream_rt,
                  TRTModel &model_be, cudaStream_t stream_be)
{
    srand(time(NULL));

    bool begin = false;
    int64_t be_cnt = 0;

    LoopRunner runner;
    runner.Start([&]() -> void {
        if (begin) be_cnt++;
        model_be.Infer(stream_be);
    });

    DataProcessor<int64_t> latency;
    std::this_thread::sleep_for(std::chrono::seconds(2));

    begin = true;
    double test_time = EXEC_TIME(nanoseconds, {
        for (int i = 0; i < LATENCY_TEST_CNT; ++i) {
            auto next = std::chrono::system_clock::now() +
                        std::chrono::milliseconds(30 + rand() % 40);
            latency.Add(EXEC_TIME(nanoseconds, {
                model_rt.Infer(stream_rt);
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

int main(int argc, char **argv)
{
    if (argc < 6) {
        XINFO("usage: %s {model dir} {input tensor dir} "
              "{model name} {batch size} {result dir}", argv[0]);
        XERRO("lack arguments, abort...");
    }

    const std::string model_dir(argv[1]);
    const std::string tensor_dir(argv[2]);
    const std::string model_name(argv[3]);
    const int batch_size = atoi(argv[4]);
    const std::string result_dir(argv[5]);
    TRTModel model_rt(model_dir + "/" + model_name + ".onnx",
                      model_dir + "/" + model_name + ".engine",
                      batch_size);
    TRTModel model_be(model_dir + "/" + model_name + ".onnx",
                      model_dir + "/" + model_name + ".engine",
                      batch_size);

    cudaStream_t stream_rt;
    cudaStream_t stream_be;
    CUDART_ASSERT(cudaStreamCreate(&stream_rt));
    CUDART_ASSERT(cudaStreamCreate(&stream_be));

    uint64_t xq_rt = CudaXQueueCreate(stream_rt, PREEMPT_MODE_STOP_SUBMISSION, 64, 32);
    uint64_t xq_be = CudaXQueueCreate(stream_be, PREEMPT_MODE_DEACTIVATE, 16, 8);

    WarmUp(model_rt, stream_rt);
    WarmUp(model_be, stream_be);

    CudaXQueueSetPriority(stream_rt, 2);
    CudaXQueueSetPriority(stream_be, 1);

    CoRun(model_rt, stream_rt, model_be, stream_be);

    CudaXQueueDestroy(xq_rt);
    CudaXQueueDestroy(xq_be);
}
