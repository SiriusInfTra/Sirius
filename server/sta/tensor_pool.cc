#include "tensor_pool.h"
#include "cuda_allocator.h"

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

TensorContainer::TensorContainer(const CUDAMemPool::PoolEntry &mdata, std::vector<int64_t> shape, DLDataType dtype) 
    : mdata_{mdata} {
  shape_ = std::move(shape);
  tensor_ = DLTensor{
    .data = mdata.addr,
    .device = DLDevice{DLDeviceType::kDLCUDA, 0},
    .ndim = static_cast<int32_t>(shape_.size()),
    .dtype = dtype,
    .shape = shape_.data(),
    .strides = nullptr,
    .byte_offset = 0
  };
} 

TensorContainer::~TensorContainer() {
  CUDAMemPool::Get()->Free(mdata_);
}

TensorPool::TensorPool() : handle_counter_{1} {
}

uint64_t TensorPool::Empty(std::vector<int64_t> shape, DLDataType dtype) {
  auto entry = CUDAMemPool::Get()->Alloc(shape.size() * dtype.bits / 8);
  if (entry.addr == nullptr) {
    return 0;
  }
  std::unique_lock lock{mutex_};
  auto handle = handle_counter_.fetch_add(1, std::memory_order_relaxed);
  tensor_by_handle_.emplace(handle, STensor(entry, shape, dtype));
  return handle;
}

void TensorPool::Free(uint64_t handle) {
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
