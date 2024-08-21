#include <common/log_as_glog_sta.h>
#include <common/tensor/tensor.h>
#include <common/cuda_allocator.h>
#include <common/tensor/shape_helper.h>


namespace colserve {
namespace sta {

TensorContainer::TensorContainer() 
    : tensor_{0}, mdata_{0}, stensor_version_{0} {}

TensorContainer::TensorContainer(std::vector<int64_t> shape, 
                                 DLDevice device, DLDataType dltype) 
    : is_null_{true}, stensor_version_{0} {
  SetTensor(nullptr, std::move(shape), device, dltype, 0);
}

TensorContainer::TensorContainer(
    std::vector<int64_t> shape, std::vector<int64_t> stride, 
    DLDevice device, DLDataType dtype, size_t storage_offset) 
    : is_null_{true}, stensor_version_{0} {
  SetTensor(nullptr, std::move(shape), std::move(stride), 
            device, dtype, storage_offset);
}

TensorContainer::TensorContainer(
    memory_data_t mdata, std::vector<int64_t> shape, 
    DLDevice device, DLDataType dtype)
    : stensor_version_{0} {
  SetTensor(mdata, std::move(shape), device, dtype, 0);
}

TensorContainer::TensorContainer(
    memory_data_t mdata, std::vector<int64_t> shape, 
    MemoryFormat memory_format, 
    DLDevice device, DLDataType dtype) {
  switch (memory_format) {
    case MemoryFormat::Contiguous:
      SetTensor(mdata, std::move(shape), device, dtype, 0);
      break;
      LOG(FATAL) << "unknown memory_format: " 
                 << static_cast<int64_t>(memory_format) << ".";
      break;
  }
}


TensorContainer::TensorContainer(
    memory_data_t mdata, std::vector<int64_t> shape, 
    std::vector<int64_t> stride, DLDevice device, 
    DLDataType dtype, size_t storage_offset)
    : stensor_version_{0} {
  SetTensor(mdata, std::move(shape), std::move(stride), 
            device, dtype, storage_offset);
}

TensorContainer::~TensorContainer() {
  // CUDAMemPool::Get()->Free(mdata_);
}

void TensorContainer::SetTensor(
    TensorContainer::memory_data_t mdata, std::vector<int64_t> shape, 
    DLDevice device, DLDataType dtype, std::optional<size_t> storage_offset) {
  std::vector<int64_t> stride(shape.size());
  if (shape.size() > 0) {
    stride.rbegin()[0] = 1;
    for (int64_t i = static_cast<int64_t>(stride.size()) - 2; i >= 0; i--) {
      stride[i] = stride[i + 1] * shape[i + 1];
    }
  }
  SetTensor(mdata, std::move(shape), std::move(stride), 
            device, dtype, storage_offset);
}

void TensorContainer::SetTensor(
    TensorContainer::memory_data_t mdata, 
    std::vector<int64_t> shape, std::vector<int64_t> stride, 
    DLDevice device, DLDataType dtype, 
    std::optional<size_t> storage_offset) {
  CHECK(device.device_type == kDLCUDA);
  mdata_ = mdata;
  shape_ = std::move(shape);
  stride_ = std::move(stride);

  size_t old_byte_offset = tensor_.byte_offset;
  tensor_ = DLTensor{
    .data = mdata ? mdata->addr : nullptr,
    .device = device,
    .ndim = static_cast<int32_t>(shape_.size()),
    .dtype = dtype,
    .shape = shape_.data(),
    .strides = stride_.data(),
    .byte_offset = storage_offset.has_value() 
                   ? storage_offset.value() * GetDataTypeNbytes(dtype) 
                   : old_byte_offset
                  //  storage_offset.value() * (dtype.bits >> 3) : old_byte_offset
  };
}

bool STensor::IsNull() const {
  CHECK_EQ(get()->is_null_, get()->mdata_ == nullptr);
  return get()->is_null_;
}

bool STensor::ComputeContiguous() const {
  bool is_contiguous = true;
  if (ComputeNumel() == 0)
    return is_contiguous;
  int64_t z = 1;
  for (int64_t d = get()->tensor_.ndim - 1; d >= 0; d--) {
    // const auto size_d = sizes_and_strides_.size_at_unchecked(d);
    const auto size_d = get()->tensor_.shape[d];
    if (size_d != 1) {
      auto stride_d = get()->tensor_.strides[d];
      if (stride_d == z) {
        z *= size_d;
      } else {
        is_contiguous = false;
        break;
      }
    }
  }
  return is_contiguous;
}

size_t STensor::ComputeNumel() const {
  int64_t numel = 1;
  for (auto dim : Shape()) {
    numel *= dim;
  }
  return numel;
}

size_t STensor::ComputeNbytes() const {
  return ComputeStorageNbytes(
      Shape(), get()->tensor_.dtype, StorageOffset());
}

}
}
