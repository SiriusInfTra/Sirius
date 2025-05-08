#pragma once

#include <map>
#include <memory>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>

#include "tensor.h"

class TRTLogger: public nvinfer1::ILogger           
{
public:
    TRTLogger() = default;
    virtual ~TRTLogger() = default;
    virtual void log(Severity severity, const char *msg) noexcept override;
};

class TRTModel
{
public:
    TRTModel(const std::string &onnx,
             const std::string &engine,
             const size_t batch_size);
    ~TRTModel() = default;

    void CopyInput(cudaStream_t stream);
    void CopyOutput(cudaStream_t stream);
    void CopyInputAsync(cudaStream_t stream);
    void CopyOutputAsync(cudaStream_t stream);

    void Infer(cudaStream_t stream);
    void InferAsync(cudaStream_t stream);

    bool CheckOutput();
    void ClearOutput(cudaStream_t stream);

    std::map<std::string, std::shared_ptr<Tensor>> &InputTensors();
    std::map<std::string, std::shared_ptr<Tensor>> &OutputTensors();

private:
    void BuildEngine(const std::string &onnx);
    void LoadEngine(const std::string &engine);
    void SaveEngine(const std::string &engine);

    void InitContext();
    void Enqueue(cudaStream_t stream);

    TRTLogger logger_;
    const int batch_size_;
    const int default_dim_size_ = 512;
    std::vector<void *> bindings_;
    nvinfer1::ICudaEngine *engine_ = nullptr;
    nvinfer1::IExecutionContext *ctx_ = nullptr;
    std::map<std::string, std::shared_ptr<Tensor>> input_tensors_;
    std::map<std::string, std::shared_ptr<Tensor>> output_tensors_;
};
