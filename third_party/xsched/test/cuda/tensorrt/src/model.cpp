#include <unistd.h>
#include <numeric>
#include <fstream>
#include <iostream>

#include "utils.h"
#include "model.h"
#include "utils/log.h"
#include "utils/xassert.h"

void TRTLogger::log(Severity severity, const char *msg) noexcept
{
    switch (severity)
    {
    case Severity::kINTERNAL_ERROR:
    case Severity::kERROR:
        XERRO("[TRT] %s", msg);
        break;
        
    case Severity::kWARNING:
        XWARN("[TRT] %s", msg);
        break;

    default:
        // suppress info-level messages
        break;
    }
}

TRTModel::TRTModel(const std::string &onnx,
                   const std::string &engine,
                   const size_t batch_size)
    : batch_size_(batch_size)
{
    if (access(engine.c_str(), F_OK) != -1) {
        // the engine file exists
        LoadEngine(engine);
    } else {
        // otherwise, build and save the engine
        BuildEngine(onnx);
        SaveEngine(engine);
    }
    InitContext();
    FLUSH_XLOG();
}

void TRTModel::CopyInput(cudaStream_t stream)
{
    CopyInputAsync(stream);
    CUDART_ASSERT(cudaStreamSynchronize(stream));
}

void TRTModel::CopyOutput(cudaStream_t stream)
{
    CopyOutputAsync(stream);
    CUDART_ASSERT(cudaStreamSynchronize(stream));
}

void TRTModel::CopyInputAsync(cudaStream_t stream)
{
    for (auto tensor : input_tensors_) {
        tensor.second->CopyToDeviceAsync(stream);
    }
}

void TRTModel::CopyOutputAsync(cudaStream_t stream)
{
    for (auto tensor : output_tensors_) {
        tensor.second->CopyToHostAsync(stream);
    }
}

void TRTModel::Infer(cudaStream_t stream)
{
    InferAsync(stream);
    CUDART_ASSERT(cudaStreamSynchronize(stream));
}

void TRTModel::InferAsync(cudaStream_t stream)
{
    Enqueue(stream);
}

bool TRTModel::CheckOutput()
{
    bool pass = true;
    for (auto out : output_tensors_) {
        if (out.second->CheckCorrect()) continue;
        pass = false;
        XWARN("[RESULT] [FAIL] output tensor %s does not match",
              out.first.c_str());
    }
    return pass;
}

void TRTModel::ClearOutput(cudaStream_t stream)
{
    for (auto out : output_tensors_) {
        out.second->Clear(stream);
    }
    CUDART_ASSERT(cudaStreamSynchronize(stream));
}

std::map<std::string, std::shared_ptr<Tensor>> &TRTModel::InputTensors()
{
    return input_tensors_;
}

std::map<std::string, std::shared_ptr<Tensor>> &TRTModel::OutputTensors()
{
    return output_tensors_;
}

void TRTModel::BuildEngine(const std::string &onnx)
{
    auto builder = nvinfer1::createInferBuilder(logger_);
    XASSERT(builder, "Failed to create InferBuilder");

    const auto explicit_batch = 1U << 
        (uint32_t)nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH;
    auto network = builder->createNetworkV2(explicit_batch);
    XASSERT(network, "Failed to create NetworkDefinition");

    auto parser = nvonnxparser::createParser(*network, logger_);
    XASSERT(parser, "Failed to create Parser");
    XASSERT(parser->parseFromFile(onnx.c_str(), false),
            "Failed to parse ONNX file");

    auto config = builder->createBuilderConfig();
    XASSERT(config, "Failed to create BuilderConfig");
    config->setMaxWorkspaceSize(2 << 20);

    auto profile = builder->createOptimizationProfile();
    XASSERT(profile, "Failed to create OptimizationProfile");

    for (int32_t i = 0; i < network->getNbInputs(); ++i) {
        nvinfer1::ITensor *input_tensor = network->getInput(i);
        nvinfer1::Dims dims = input_tensor->getDimensions();

        bool dynamic_shape = false;
        if (dims.d[0] == -1) {
            dims.d[0] = batch_size_;
            dynamic_shape = true;
        }
        for (int32_t j = 1; j < dims.nbDims; ++j) {
            if (dims.d[j] == -1) {
                dims.d[j] = default_dim_size_;
                dynamic_shape = true;
            }
        }

        if (dynamic_shape) {
            profile->setDimensions(input_tensor->getName(),
                                   nvinfer1::OptProfileSelector::kMIN,
                                   dims);
            profile->setDimensions(input_tensor->getName(),
                                   nvinfer1::OptProfileSelector::kOPT,
                                   dims);
            profile->setDimensions(input_tensor->getName(),
                                   nvinfer1::OptProfileSelector::kMAX,
                                   dims);
        }
    }

    config->addOptimizationProfile(profile);
    engine_ = builder->buildEngineWithConfig(*network, *config);
    XINFO("[MODEL] engine built with model %s", onnx.c_str());
}

void TRTModel::LoadEngine(const std::string &engine)
{
    std::ifstream engine_file(engine, std::ios::binary | std::ios::ate);
    std::streamsize size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    XASSERT(engine_file.read(buffer.data(), size),
            "Failed to read engine file");

    auto runtime = nvinfer1::createInferRuntime(logger_);
    XASSERT(runtime, "Failed to create InferRuntime");

    bool init_plugins = initLibNvInferPlugins(&logger_, "");
    engine_ = runtime->deserializeCudaEngine(buffer.data(), size);
    XASSERT(engine_, "Failed to deserialize CUDA engine");

    engine_file.close();
    XINFO("[MODEL] engine %s loaded", engine.c_str());
}

void TRTModel::SaveEngine(const std::string &engine)
{
    auto mem = engine_->serialize();

    std::ofstream engine_file(engine, std::ios::app);
    engine_file.write((const char *)mem->data(), mem->size());
    engine_file.close();
    XINFO("[MODEL] engine %s saved", engine.c_str());
}

void TRTModel::InitContext()
{
    ctx_ = engine_->createExecutionContext();
    XASSERT(ctx_, "Failed to create IExecutionContext");

    for (int32_t i = 0; i < engine_->getNbBindings(); i++) {
        bool dynamic_shape = false;
        nvinfer1::Dims dims = engine_->getBindingDimensions(i);
        if (dims.d[0] == -1) {
            dims.d[0] = batch_size_;
            dynamic_shape = true;
        }
        for (int32_t j = 1; j < dims.nbDims; ++j) {
            if (dims.d[j] == -1) {
                dims.d[j] = default_dim_size_;
                dynamic_shape = true;
            }
        }

        if (dynamic_shape && engine_->getNbOptimizationProfiles() > 0) {
            dims = engine_->getProfileDimensions(
                i, 0, nvinfer1::OptProfileSelector::kOPT);
        }

        nvinfer1::DataType data_type = engine_->getBindingDataType(i);
        int32_t volume = std::accumulate(dims.d, dims.d + dims.nbDims, 1,
                                         std::multiplies<int32_t>());
        size_t total_size = volume * GetSize(data_type);

        bool is_input = engine_->bindingIsInput(i);
        std::string tensor_name(engine_->getBindingName(i));
        std::shared_ptr<Tensor> tensor = std::make_shared<Tensor>(total_size,
                                                                  dims,
                                                                  tensor_name);

        std::string log_str;
        if (is_input) {
            log_str = "[MODEL] created input tensor, name: "
                    + tensor_name + ", "
                    + GetTypeName(data_type)
                    + "[" + std::to_string(dims.d[0]);
            input_tensors_[tensor_name] = tensor;
        } else {
            log_str = "[MODEL] created output tensor, name: "
                    + tensor_name + ", "
                    + GetTypeName(data_type)
                    + "[" + std::to_string(dims.d[0]);
            output_tensors_[tensor_name] = tensor;
        }
        
        for (int32_t j = 1; j < dims.nbDims; ++j) {
            log_str += ", " + std::to_string(dims.d[j]);
        }
        log_str += "] (" + std::to_string(total_size) + " Bytes)";
        XINFO("%s", log_str.c_str());

        bindings_.emplace_back(tensor->DeviceBuffer());
        if (dynamic_shape && is_input) {
            ctx_->setBindingDimensions(i, dims);
        }
    }

    XINFO("[MODEL] infer context initialized");
}

void TRTModel::Enqueue(cudaStream_t stream)
{
    XASSERT(ctx_->enqueueV2(bindings_.data(), stream, nullptr),
            "Failed to enqueue");
}
