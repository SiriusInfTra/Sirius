#include "logging_as_glog.h"
#include "tensor_pool.h"
#include "cuda_allocator.h"
#include "shape_helper.h"


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


TensorContainer::TensorContainer() : tensor_{0}, mdata_{0}, stensor_version_{0} {}

TensorContainer::TensorContainer(std::vector<int64_t> shape, DLDataType dltype) 
    : is_null_{true}, stensor_version_{0} {
  SetTensor(nullptr, std::move(shape), dltype, 0);
}

TensorContainer::TensorContainer(
    std::vector<int64_t> shape, std::vector<int64_t> stride, 
    DLDataType dtype, size_t storage_offset) 
    : is_null_{true}, stensor_version_{0} {
  SetTensor(nullptr, std::move(shape), std::move(stride), dtype, storage_offset);
}

TensorContainer::TensorContainer(memory_data_t mdata, std::vector<int64_t> shape, DLDataType dtype)
    : stensor_version_{0} {
  SetTensor(mdata, std::move(shape), dtype, 0);
}

TensorContainer::TensorContainer(memory_data_t mdata, std::vector<int64_t> shape, at::MemoryFormat memory_format, 
                  DLDataType dtype) {
  switch (memory_format) {
    case c10::MemoryFormat::Contiguous:
      SetTensor(mdata, std::move(shape), dtype, 0);
      break;
    case c10::MemoryFormat::ChannelsLast:
      SetTensor(mdata, std::move(shape), c10::get_channels_last_strides_2d(shape), dtype, 0);
      break;
    case c10::MemoryFormat::ChannelsLast3d:
      SetTensor(mdata, std::move(shape), c10::get_channels_last_strides_2d(shape), dtype, 0);
    default:
      LOG(FATAL) << "unknown memory_format: " << memory_format << ".";
      break;
  }
}


TensorContainer::TensorContainer(
    memory_data_t mdata, std::vector<int64_t> shape, std::vector<int64_t> stride,
    DLDataType dtype, size_t storage_offset)
    : stensor_version_{0} {
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
                   storage_offset.value() * GetDataTypeNbytes(dtype) : old_byte_offset
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
  if (train_model_allocating_) {
    train_model_tensor_handles_.insert(handle);
  }
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
  std::unique_lock lock{mutex_};
  // auto tensor = tensor_by_handle_.at(handle);
  // return tensor_by_handle_.at(handle);
  auto it = tensor_by_handle_.find(handle);
  CHECK(it != tensor_by_handle_.end()) << "TensorPool: handle " << handle << " not found";
  return it->second;
}

const STensor TensorPool::CTensor(uint64_t handle) {
  return Tensor(handle);
}

void TensorPool::AddTrainIntermediateTensor(uint64_t handle) {
  std::unique_lock lock{mutex_};
  auto it = tensor_by_handle_.find(handle);
  // CHECK(tensor_by_handle_.count(handle)) << "TensorPool: handle " << handle << " not found";
  CHECK(it != tensor_by_handle_.end()) << "TensorPool: handle " << handle << " not found";
  train_intermediate_tensor_handles_.push_back(handle);
  train_intermediate_tensor_memory_.insert(it->second.MData()->addr);
}

void TensorPool::ClearTrainIntermediateTensor() {
  std::unique_lock lock{mutex_};
  train_intermediate_tensor_handles_.clear();
  train_intermediate_tensor_memory_.clear();
}

void TensorPool::ReleaseTrainIntermediateTensorMemory() {
  std::unique_lock lock{mutex_};
  auto t0 = std::chrono::steady_clock::now();
  for (auto it : tensor_by_handle_) {
    if (train_model_tensor_handles_.count(it.first)) continue; // skip model parameters
    if (train_intermediate_tensor_memory_.count(it.second.MData()->addr)) {
      DLOG(INFO) << "ReleaseTrainIntermediateTensorMemory: " << it.first << " " << it.second.MData()->addr;
      it.second.DeallocToDummy();
    }
  }
  auto t1 = std::chrono::steady_clock::now();
  // LOG(INFO) << "ReleaseTrainIntermediateTensorMemory: " 
  //           << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms";
}

void TensorPool::RearrangeTrainMemory() {
  std::unique_lock lock{mutex_};
  auto t0 = std::chrono::steady_clock::now();
  for (auto it : tensor_by_handle_) {
    if (train_model_tensor_handles_.count(it.first)) continue; // skip model parameters
    it.second.Rearrange();
  }
  auto t1 = std::chrono::steady_clock::now();
  LOG(INFO) << "RearrangeTrainMemory: " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms";
}

}
}
