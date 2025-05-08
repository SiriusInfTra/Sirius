#include "test.h"
#include "utils.h"
#include "model.h"
#include "utils/log.h"
#include "utils/data.h"
#include "utils/psync.h"
#include "utils/timing.h"
#include "utils/runner.h"

#include <cuda_runtime.h>

using namespace xsched::utils;

static void RtTask(TRTModel &model, cudaStream_t stream)
{
    srand(time(NULL));

    ProcessSync psync;
    DataProcessor<int64_t> latency;

    WarmUp(model, stream);

    psync.Sync(2, "rt");

    int64_t ns = EXEC_TIME(nanoseconds, {
        for (int64_t i = 0; i < LATENCY_TEST_CNT; ++i) {
            auto next = std::chrono::system_clock::now() +
                        std::chrono::milliseconds(30 + rand() % 40);
            latency.Add(EXEC_TIME(nanoseconds, {
                model.Infer(stream);
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

static void BeTask(TRTModel &model, cudaStream_t stream)
{
    ProcessSync psync;

    WarmUp(model, stream);

    psync.Sync(2, "be");

    int64_t infer_cnt = 0;
    int64_t ns = EXEC_TIME(nanoseconds, {
        while (psync.GetCnt() < 3) {
            model.Infer(stream);
            infer_cnt++;
        }
    });

    XINFO("[RESULT] be throughput %.2f reqs/s",
          (double)infer_cnt * 1e9 / ns);
}

int main(int argc, char **argv)
{
    if (argc < 6) {
        XINFO("usage: %s {model dir} {input tensor dir} "
              "{model name} {batch size} {result dir}", argv[0]);
        XERRO("lack arguments, abort...");
    }

    // parent is rt task
    pid_t cpid = fork();

    const std::string model_dir(argv[1]);
    const std::string tensor_dir(argv[2]);
    const std::string model_name(argv[3]);
    const int batch_size = atoi(argv[4]);
    const std::string result_dir(argv[5]);
    TRTModel model(model_dir + "/" + model_name + ".onnx",
                   model_dir + "/" + model_name + ".engine",
                   batch_size);

    cudaStream_t stream;
    CUDART_ASSERT(cudaStreamCreate(&stream));

    if (cpid) {
        // parent rt task
        RtTask(model, stream);
    } else {
        // child be task
        BeTask(model, stream);
    }
}
