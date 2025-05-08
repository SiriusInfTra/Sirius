#include <fstream>

#include "utils.h"
#include "tensor.h"
#include "utils/log.h"
#include "utils/xassert.h"

void FillRandom(void *buf, size_t size)
{
    srand(time(0));
    char *buf1 = (char *)buf;
    int *buf4 = (int *)buf;
    size_t size4 = size / sizeof(int);
    for (size_t i = 0; i < size4; ++i) buf4[i] = rand();
    for (size_t i = size4 * sizeof(int); i < size; ++i) buf1[i] = rand();
}

size_t GetSize(nvinfer1::DataType data_type)
{
    switch (data_type)
    {
    case nvinfer1::DataType::kFLOAT:
        return 4;
    
    case nvinfer1::DataType::kHALF:
        return 2;

    case nvinfer1::DataType::kINT8:
        return 1;

    case nvinfer1::DataType::kINT32:
        return 4;
    
    case nvinfer1::DataType::kBOOL:
        return 1;
    
    default:
        XASSERT(false, "unknown data type(%d)", (int)data_type);
        return 8;
    }
}

const char *GetTypeName(nvinfer1::DataType data_type)
{
    switch (data_type)
    {
    case nvinfer1::DataType::kFLOAT:
        return "FLOAT";
    
    case nvinfer1::DataType::kHALF:
        return "HALF";

    case nvinfer1::DataType::kINT8:
        return "INT8";

    case nvinfer1::DataType::kINT32:
        return "INT32";
    
    case nvinfer1::DataType::kBOOL:
        return "BOOL";
    
    default:
        XASSERT(false, "unknown data type(%d)", (int)data_type);
        return "UNKNOWN";
    }
}

Tensor::Tensor(size_t size,
               const nvinfer1::Dims &dims,
               const std::string &name)
    : size_(size), dims_(dims), name_(name)
{
    correct_buf_ = malloc(size_);
    host_buf_ = malloc(size_);
    CUDART_ASSERT(cudaMalloc(&device_buf_, size_));
}

Tensor::~Tensor()
{
    CUDART_ASSERT(cudaFree(device_buf_));
}

void Tensor::CopyToHostAsync(cudaStream_t stream)
{
    CUDART_ASSERT(cudaMemcpyAsync(host_buf_, device_buf_, size_,
                                  cudaMemcpyDeviceToHost, stream));
}

void Tensor::CopyToDeviceAsync(cudaStream_t stream)
{
    CUDART_ASSERT(cudaMemcpyAsync(device_buf_, host_buf_, size_,
                                  cudaMemcpyHostToDevice, stream));
}

void Tensor::CopyTo(void *dst)
{
    memcpy(dst, host_buf_, size_);
}

void Tensor::CopyFrom(void *src)
{
    memcpy(host_buf_, src, size_);
}

void Tensor::Clear(cudaStream_t stream)
{
    CUDART_ASSERT(cudaMemsetAsync(device_buf_, 0, size_, stream));
}

void Tensor::SaveCorrect()
{
    memcpy(correct_buf_, host_buf_, size_);
}

bool Tensor::CheckCorrect()
{
    return memcmp(correct_buf_, host_buf_, size_) == 0;
}

bool Tensor::Compare(const void *buf)
{
    return memcmp(host_buf_, buf, size_) == 0;
}

void Tensor::Load(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.good()) {
        file.close();
        FillRandom(host_buf_, size_);
        XWARN("[TENSOR] input tensor file %s does not exist, "
              "tensor %s will be randoms", filename.c_str(), name_.c_str());
        return;
    }

    std::streamsize size = file.tellg();
    XASSERT(size == size_,
            "file size (%ld Bytes) mismatch tensor size (%ld Bytes)",
            size, size_);
    XINFO("[TENSOR] load %s to tensor %s (%ld Bytes)",
          filename.c_str(), name_.c_str(), size_);
    
    file.seekg(0, std::ios::beg);
    XASSERT(file.read((char *)host_buf_, size),
            "read file %s failed", filename.c_str());
    file.close();
}

void Tensor::Save(const std::string &filename)
{
    XINFO("[TENSOR] save tensor %s (%ld Bytes) to %s",
          name_.c_str(), size_, filename.c_str());
    std::ofstream file(filename, std::ios::binary);
    file.write((const char *)host_buf_, size_);
    file.close();
}
