#include "test.h"
#include "utils.h"
#include "model.h"
#include "utils/log.h"
#include "shim/cuda/xctrl.h"

#include <algorithm>
#include <cuda_runtime.h>

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
    TRTModel model(model_dir + "/" + model_name + ".onnx",
                   model_dir + "/" + model_name + ".engine",
                   batch_size);

    cudaStream_t stream;
    CUDART_ASSERT(cudaStreamCreate(&stream));
    
    // cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

    XINFO("[TEST] xsched tensorrt preempt test for %s", model_name.c_str());
    // save correct results
    for (auto tensor : model.InputTensors()) {
        std::string tensor_name = tensor.first;
        std::replace(tensor_name.begin(), tensor_name.end(), '/', '-');
        tensor.second->Load(tensor_dir + "/"
            + model_name + "_" + tensor_name + ".bin");
    }
    
    model.CopyInputAsync(stream);
    model.InferAsync(stream);
    model.CopyOutput(stream);
    CUDART_ASSERT(cudaStreamSynchronize(stream));

    for (auto tensor : model.OutputTensors()) {
        tensor.second->SaveCorrect();
    }

    XINFO("[TEST] inference baseline test");
    WarmUp(model, stream);
    int64_t disable_ns = InferLatencyNs(model, stream);
    XINFO("[RESULT] disable preemption: %ldus", disable_ns / 1000);

    XINFO("[TEST] inference overhead test");
    XINFO("[TEST] CudaXQueueCreate()");
    uint64_t handle = CudaXQueueCreate(stream, PREEMPT_MODE_INTERRUPT, 64, 16);
    // launch to make sure that all kernels are instrumented
    WarmUp(model, stream);
    int64_t enable_ns = InferLatencyNs(model, stream);
    XINFO("[RESULT] enable preemption: %ldus", enable_ns / 1000);
    XINFO("[RESULT] overhead: %ldus, %.2f%%",
          (enable_ns - disable_ns) / 1000,
          double(enable_ns - disable_ns) / double(disable_ns) * 100);

    XINFO("[TEST] inference preempt test");
    PreemptTest(model, stream);

    XINFO("[TEST] inference correctness test");
    CorrectnessTest(model, stream);

    XINFO("[TEST] test complete");
    XINFO("[TEST] CudaXQueueDestroy()");
    CudaXQueueDestroy(handle);
    CUDART_ASSERT(cudaStreamDestroy(stream));
    return 0;
}
