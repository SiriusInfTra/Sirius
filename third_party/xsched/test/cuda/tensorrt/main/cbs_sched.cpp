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

static void CoRun(TRTModel &model1, cudaStream_t stream1,
                  TRTModel &model2, cudaStream_t stream2)
{
    DataProcessor<int64_t> latency1;
    DataProcessor<int64_t> latency2;

    bool begin = false;

    LoopRunner runner;
    runner.Start([&]() -> void {
        int64_t exec_time = EXEC_TIME(nanoseconds, {
            model2.Infer(stream2);
        });
        if (begin) latency2.Add(exec_time);
    });

    std::this_thread::sleep_for(std::chrono::seconds(2));

    begin = true;
    double test_time = EXEC_TIME(nanoseconds, {
        for (int i = 0; i < LATENCY_TEST_CNT; ++i) {
            latency1.Add(EXEC_TIME(nanoseconds, {
                model1.Infer(stream1);
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
    TRTModel model1(model_dir + "/" + model_name + ".onnx",
                    model_dir + "/" + model_name + ".engine",
                    batch_size);
    TRTModel model2(model_dir + "/" + model_name + ".onnx",
                    model_dir + "/" + model_name + ".engine",
                    batch_size);

    cudaStream_t stream1;
    cudaStream_t stream2;
    CUDART_ASSERT(cudaStreamCreate(&stream1));
    CUDART_ASSERT(cudaStreamCreate(&stream2));

    uint64_t xq1 = CudaXQueueCreate(stream1, PREEMPT_MODE_STOP_SUBMISSION, 16, 8);
    uint64_t xq2 = CudaXQueueCreate(stream2, PREEMPT_MODE_STOP_SUBMISSION, 16, 8);

    WarmUp(model1, stream1);
    WarmUp(model2, stream2);

    CudaXQueueSetBandwidth(stream1, 3);
    CudaXQueueSetBandwidth(stream2, 1);

    CoRun(model1, stream1, model2, stream2);

    CudaXQueueDestroy(xq1);
    CudaXQueueDestroy(xq2);
}
