#include "tensor_pool.h"
#include "cuda_allocator.h"
#include "shape_helper.h"

namespace colserve {
namespace sta {

std::unique_ptr<TensorPool> TensorPool::tensor_pool_ = nullptr;

void TensorPool::Init() {
  std::cout << "initlize TensorPool" << std::endl;
  tensor_pool_ = std::make_unique<TensorPool>();
}

TensorPool* TensorPool::Get() {
  return tensor_pool_.get();
}

TensorContainer::TensorContainer() : tensor_{}, mdata_{} {}

TensorContainer::TensorContainer(memory_data_t mdata, std::vector<int64_t> shape, DLDataType dtype) 
    : mdata_{mdata} {
  shape_ = std::move(shape);
  stride_.resize(shape_.size());
  stride_.rbegin()[0] = 1;
  for (int64_t i = static_cast<int64_t>(stride_.size()) - 2; i >= 0; i--) {
    stride_[i] = stride_[i + 1] * shape_[i + 1];
  }
  tensor_ = DLTensor{
    .data = mdata->addr,
    .device = DLDevice{DLDeviceType::kDLCUDA, 0},
    .ndim = static_cast<int32_t>(shape_.size()),
    .dtype = dtype,
    .shape = shape_.data(),
    .strides = stride_.data(),
    .byte_offset = 0
  };
}

TensorContainer::TensorContainer(
    memory_data_t mdata, std::vector<int64_t> shape, std::vector<int64_t> stride,
    DLDataType dtype, size_t storage_offset) 
    : mdata_{mdata} {
  shape_ = std::move(shape);
  stride_ = std::move(stride);
  tensor_ = DLTensor{
    .data = mdata->addr,
    .device = DLDevice{DLDeviceType::kDLCUDA, 0},
    .ndim = static_cast<int32_t>(shape_.size()),
    .dtype = dtype,
    .shape = shape_.data(),
    .strides = stride_.data(),
    .byte_offset = storage_offset * dtype.bits / 8
  };
}

TensorContainer::~TensorContainer() {
  // CUDAMemPool::Get()->Free(mdata_);
}

// STensor STensor::AsStrided(const std::vector<int64_t> &size, 
//                            const std::vector<int64_t> &stride,
//                            int64_t storage_offset) const {
//   CheckMemoryBound(size, stride, get()->tensor_.dtype, storage_offset, get()->mdata_);
//   return STensor(get()->mdata_, size, stride, get()->tensor_.dtype, storage_offset);
// }

TensorPool::TensorPool() : handle_counter_{1} {
}

uint64_t TensorPool::Insert(STensor tensor) {
  std::unique_lock lock{mutex_};
  auto handle = handle_counter_.fetch_add(1, std::memory_order_relaxed);
  tensor_by_handle_.emplace(handle, tensor);
  return handle;
} 

// uint64_t TensorPool::Empty(std::vector<int64_t> shape, DLDataType dtype) {
//   auto entry = CUDAMemPool::Get()->Alloc(shape.size() * dtype.bits / 8);
//   if (entry == nullptr) {
//     return 0;
//   }
//   std::unique_lock lock{mutex_};
//   auto handle = handle_counter_.fetch_add(1, std::memory_order_relaxed);
//   tensor_by_handle_.emplace(handle, STensor(entry, shape, dtype));
//   return handle;
// }

void TensorPool::Remove(uint64_t handle) {
  std::unique_lock lock{mutex_};
  tensor_by_handle_.erase(handle);
}

STensor TensorPool::Tensor(uint64_t handle) {
  // auto tensor = tensor_by_handle_.at(handle);
  return tensor_by_handle_.at(handle);
}

const STensor TensorPool::CTensor(uint64_t handle) const {
  return tensor_by_handle_.at(handle);
}


}
}
