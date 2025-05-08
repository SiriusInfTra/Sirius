#pragma once

#include <list>
#include <string>
#include <memory>
#include <unordered_map>

#include <cudla.h>
#include <cuda_runtime.h>

struct CudlaTensor
{
    size_t size;
    void *addr_gpu;
    std::string name;
};

struct CudlaLayer
{
    size_t size;
    char *loadable;
    std::string name;
    cudlaDevHandle dla;
    cudlaModule module;
    uint32_t num_input_tensors;
    uint32_t num_output_tensors;
    void **input_addrs;
    void **output_addrs;
    size_t *input_sizes;
    size_t *output_sizes;
    CudlaTensor **input_tensors;
    CudlaTensor **output_tensors;
};

class CudlaModel
{
public:
    CudlaModel(const std::string &engine_dir);
    ~CudlaModel();

    void Infer(cudaStream_t stream);
    void InferAsync(cudaStream_t stream);

private:
    std::list<CudlaLayer *> layers_;
    std::unordered_map<std::string, CudlaTensor *> tensors_;

    void LoadLayer(char *loadable, size_t size);
};
