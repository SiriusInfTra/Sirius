#include <fstream>
#include <cassert>

#include "utils.h"
#include "model.h"
#include "utils/log.h"
#include "utils/common.h"

CudlaModel::CudlaModel(const std::string &engine_dir)
{
    CUDART_ASSERT(cudaFree(0));
    CUDART_ASSERT(cudaSetDevice(0));

    while (true) {
        std::ifstream file(engine_dir + "/" + std::to_string(layers_.size())
                           + ".engine", std::ios::binary);
        if (!file.good()) break;
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        char *loadable = new char[size];
        file.read(loadable, size);
        file.close();
        XINFO("loading layer %3ld, size: %ld Bytes", layers_.size(), size);
        LoadLayer(loadable, size);
        if (layers_.size() == 16) {
            // TODO: support more layers
            break;
        }
    }
    XINFO("load done");

    for (auto layer : layers_) {
        for (uint32_t i = 0; i < layer->num_input_tensors; i++) {
            uint64_t *addr_dla = nullptr;
            CudlaTensor *tensor = layer->input_tensors[i];
            CUDLA_ASSERT(cudlaMemRegister(layer->dla,
                                          (uint64_t *)tensor->addr_gpu,
                                          layer->input_sizes[i],
                                          &addr_dla, 0));
            layer->input_addrs[i] = addr_dla;
            XINFO("register input %ld bytes at %p in %p",
                  layer->input_sizes[i], addr_dla, layer->dla);
        }

        for (uint32_t i = 0; i < layer->num_output_tensors; i++) {
            uint64_t *addr_dla = nullptr;
            CudlaTensor *tensor = layer->output_tensors[i];
            CUDLA_ASSERT(cudlaMemRegister(layer->dla,
                                          (uint64_t *)tensor->addr_gpu,
                                          layer->output_sizes[i],
                                          &addr_dla, 0));
            layer->output_addrs[i] = addr_dla;
            XINFO("register output %ld bytes at %p in %p",
                  layer->output_sizes[i], addr_dla, layer->dla);
        }
    }
}

CudlaModel::~CudlaModel()
{
    for (auto layer : layers_) {
        for (uint32_t i = 0; i < layer->num_input_tensors; i++) {
            CUDLA_ASSERT(cudlaMemUnregister(
                layer->dla, (const uint64_t *)layer->input_addrs[i]));
        }
        for (uint32_t i = 0; i < layer->num_output_tensors; i++) {
            CUDLA_ASSERT(cudlaMemUnregister(
                layer->dla, (const uint64_t *)layer->output_addrs[i]));
        }
        delete[] layer->loadable;
        delete[] layer->input_addrs;
        delete[] layer->output_addrs;
        delete[] layer->input_tensors;
        delete[] layer->output_tensors;
        CUDLA_ASSERT(cudlaModuleUnload(layer->module, 0));
        CUDLA_ASSERT(cudlaDestroyDevice(layer->dla));
        delete layer;
    }
    layers_.clear();

    for (auto tensor : tensors_) {
        CUDART_ASSERT(cudaFree(tensor.second->addr_gpu));
        delete tensor.second;
    }
    tensors_.clear();
}

void CudlaModel::Infer(cudaStream_t stream)
{
    InferAsync(stream);
    CUDART_ASSERT(cudaStreamSynchronize(stream));
}

void CudlaModel::InferAsync(cudaStream_t stream)
{
    cudlaTask task;
    task.waitEvents = NULL;
    task.signalEvents = NULL;
    for (auto layer : layers_) {
        task.moduleHandle = layer->module;
        task.numInputTensors = layer->num_input_tensors;
        task.numOutputTensors = layer->num_output_tensors;
        task.inputTensor = (uint64_t *const *)layer->input_addrs;
        task.outputTensor = (uint64_t *const *)layer->output_addrs;
        CUDLA_ASSERT(cudlaSubmitTask(layer->dla, &task, 1, stream, 0));
    }
}

void CudlaModel::LoadLayer(char *loadable, size_t size)
{
    cudlaDevHandle dla;
    cudlaModule module;
    cudlaModuleAttribute attr;
    uint32_t num_input_tensors;
    uint32_t num_output_tensors;

    CUDLA_ASSERT(cudlaCreateDevice(0, &dla, CUDLA_CUDA_DLA));
    CUDLA_ASSERT(cudlaModuleLoadFromMemory(dla, (const uint8_t *)loadable,
                                           size, &module, 0));
    CUDLA_ASSERT(cudlaModuleGetAttributes(module, CUDLA_NUM_INPUT_TENSORS,
                                          &attr));
    num_input_tensors = attr.numInputTensors;
    CUDLA_ASSERT(cudlaModuleGetAttributes(module, CUDLA_NUM_OUTPUT_TENSORS,
                                          &attr));
    num_output_tensors = attr.numOutputTensors;

    cudlaModuleTensorDescriptor *input_tensor_desc
        = new cudlaModuleTensorDescriptor[num_input_tensors];
    cudlaModuleTensorDescriptor *output_tensor_desc
        = new cudlaModuleTensorDescriptor[num_output_tensors];
    
    attr.inputTensorDesc = input_tensor_desc;
    CUDLA_ASSERT(cudlaModuleGetAttributes(
        module, CUDLA_INPUT_TENSOR_DESCRIPTORS, &attr));
    attr.outputTensorDesc = output_tensor_desc;
    CUDLA_ASSERT(cudlaModuleGetAttributes(
        module, CUDLA_OUTPUT_TENSOR_DESCRIPTORS, &attr));

    auto layer = new CudlaLayer {
        .size = size,
        .loadable = loadable,
        .name = "",
        .dla = dla,
        .module = module,
        .num_input_tensors = num_input_tensors,
        .num_output_tensors = num_output_tensors,
        .input_addrs = new void *[num_input_tensors],
        .output_addrs = new void *[num_output_tensors],
        .input_sizes = new size_t[num_input_tensors],
        .output_sizes = new size_t[num_output_tensors],
        .input_tensors = new CudlaTensor *[num_input_tensors],
        .output_tensors = new CudlaTensor *[num_output_tensors],
    };

    for (uint32_t i = 0; i < num_input_tensors; i++) {
        CudlaTensor *tensor = nullptr;
        uint64_t size = input_tensor_desc[i].size;
        std::string tensor_name(input_tensor_desc[i].name);
        XINFO("input tenosr %s: size: %ld, %d[%ld,%ld,%ld,%ld]",
              tensor_name.c_str(), size,
              input_tensor_desc[i].dataType,
              input_tensor_desc[i].n,
              input_tensor_desc[i].c,
              input_tensor_desc[i].h,
              input_tensor_desc[i].w);

        if (tensors_.find(tensor_name) != tensors_.end()) {
            tensor = tensors_[tensor_name];
            if (tensor->size < size) {
                CUDART_ASSERT(cudaFree(tensor->addr_gpu));
                void *addr_gpu = nullptr;
                CUDART_ASSERT(cudaMalloc(&addr_gpu, size));
                tensor->addr_gpu = addr_gpu;
                tensor->size = size;
            }
        } else {
            void *addr_gpu = nullptr;
            CUDART_ASSERT(cudaMalloc(&addr_gpu, size));
            tensor = new CudlaTensor {
                .size = size,
                .addr_gpu = addr_gpu,
                .name = tensor_name
            };
            tensors_[tensor_name] = tensor;
        }

        layer->input_sizes[i] = size;
        layer->input_tensors[i] = tensor;
    }

    for (uint32_t i = 0; i < num_output_tensors; i++) {
        uint64_t size = output_tensor_desc[i].size;
        std::string tensor_name(output_tensor_desc[i].name);
        assert(tensors_.find(tensor_name) == tensors_.end());
        XINFO("output tenosr %s: size: %ld, %d[%ld,%ld,%ld,%ld]",
              tensor_name.c_str(), size,
              output_tensor_desc[i].dataType,
              output_tensor_desc[i].n,
              output_tensor_desc[i].c,
              output_tensor_desc[i].h,
              output_tensor_desc[i].w);
 
        void *addr_gpu = nullptr;
        CUDART_ASSERT(cudaMalloc(&addr_gpu, size));
        
        auto tensor = new CudlaTensor {
            .size = size,
            .addr_gpu = addr_gpu,
            .name = tensor_name
        };
        tensors_[tensor_name] = tensor;
        layer->output_sizes[i] = size;
        layer->output_tensors[i] = tensor;
    }

    layers_.push_back(layer);
}
