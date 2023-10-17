#include "tensor_pool.h"
#include "cuda_allocator.h"
#include "shape_helper.h"

#include <glog/logging.h>

namespace colserve {
namespace sta {

std::unique_ptr<TensorPool> TensorPool::tensor_pool_ = nullptr;

void TensorPool::Init() {
  LOG(INFO) << "initlize TensorPool" << std::endl;
  tensor_pool_ = std::make_unique<TensorPool>();
}

TensorPool* TensorPool::Get() {
  return tensor_pool_.get();
}


TensorContainer::TensorContainer() : tensor_{0}, mdata_{0} {}

TensorContainer::TensorContainer(std::vector<int64_t> shape, DLDataType dltype) : is_null_{true} {
  SetTensor(nullptr, std::move(shape), dltype, 0);
}

TensorContainer::TensorContainer(std::vector<int64_t> shape, std::vector<int64_t> stride, 
                                 DLDataType dtype, size_t storage_offset) : is_null_{true} {
  SetTensor(nullptr, std::move(shape), std::move(stride), dtype, storage_offset);
}

TensorContainer::TensorContainer(memory_data_t mdata, std::vector<int64_t> shape, DLDataType dtype) {
  SetTensor(mdata, std::move(shape), dtype, 0);
}

TensorContainer::TensorContainer(
    memory_data_t mdata, std::vector<int64_t> shape, std::vector<int64_t> stride,
    DLDataType dtype, size_t storage_offset) {
  SetTensor(mdata, std::move(shape), std::move(stride), dtype, storage_offset);
}

TensorContainer::~TensorContainer() {
  // CUDAMemPool::Get()->Free(mdata_);
}

void TensorContainer::SetTensor(TensorContainer::memory_data_t mdata, 
    std::vector<int64_t> shape, DLDataType dtype, std::optional<size_t> storage_offset) {
  std::vector<int64_t> stride(shape.size());
  if (shape.size() > 0) {
    stride.rbegin()[0] = 1;
    for (int64_t i = static_cast<int64_t>(stride.size()) - 2; i >= 0; i--) {
      stride[i] = stride[i + 1] * shape[i + 1];
    }
  }
  SetTensor(mdata, std::move(shape), std::move(stride), dtype, storage_offset);
}

void TensorContainer::SetTensor(TensorContainer::memory_data_t mdata, 
    std::vector<int64_t> shape, std::vector<int64_t> stride, 
    DLDataType dtype, std::optional<size_t> storage_offset) {
  mdata_ = mdata;
  shape_ = std::move(shape);
  stride_ = std::move(stride);

  size_t old_byte_offset = tensor_.byte_offset;
  tensor_ = DLTensor{
    .data = mdata ? mdata->addr : nullptr,
    .device = DLDevice{DLDeviceType::kDLCUDA, 0},
    .ndim = static_cast<int32_t>(shape_.size()),
    .dtype = dtype,
    .shape = shape_.data(),
    .strides = stride_.data(),
    .byte_offset = storage_offset.has_value() ? 
                   storage_offset.value() * (dtype.bits >> 3) : old_byte_offset
  };
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
  auto numel = 1;
  for (auto dim : Shape()) {
    numel *= dim;
  }
  return numel;
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

STensor TensorPool::Tensor(uint64_t handle) const {
  // auto tensor = tensor_by_handle_.at(handle);
  // return tensor_by_handle_.at(handle);
  auto it = tensor_by_handle_.find(handle);
  CHECK(it != tensor_by_handle_.end()) << "TensorPool: handle " << handle << " not found";
  return it->second;
}

const STensor TensorPool::CTensor(uint64_t handle) const {
  return Tensor(handle);
}


}
}
