#pragma once

#include <string>
#include <cstring>
#include <cstdlib>
#include <NvInfer.h>
#include <cuda_runtime.h>

void FillRandom(void *buf, size_t size);
size_t GetSize(nvinfer1::DataType data_type);
const char *GetTypeName(nvinfer1::DataType data_type);

class Tensor
{
public:
    Tensor(size_t size, const nvinfer1::Dims &dims, const std::string &name);
    ~Tensor();

    void *HostBuffer() { return host_buf_; }
    void *DeviceBuffer() { return device_buf_; }

    void CopyToHostAsync(cudaStream_t stream);
    void CopyToDeviceAsync(cudaStream_t stream);

    void CopyTo(void *dst);
    void CopyFrom(void *src);
    void Clear(cudaStream_t stream);

    void SaveCorrect();
    bool CheckCorrect();
    bool Compare(const void *buf);

    void Load(const std::string &filename);
    void Save(const std::string &filename);

private:
    const size_t size_;
    const nvinfer1::Dims dims_;
    const std::string name_;

    void *host_buf_ = nullptr;
    void *device_buf_ = nullptr;
    void *correct_buf_ = nullptr;
};
