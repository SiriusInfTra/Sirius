#include <ATen/native/TensorConversions.h>

#include "tensor_methods.h"
#include "tensor_pool.h"
#include "shape_helper.h"
#include "dtype_helper.h"

namespace colserve {
namespace sta {

uint64_t Empty(at::IntArrayRef size, DLDataType dtype) {
  auto storage_nbytes = ComputeStorageNbytes(size, dtype);
  auto entry = CUDAMemPool::Get()->Alloc(storage_nbytes);
  std::stringstream ss;
  ss << "Create empty size " << size << " nbytes " << storage_nbytes;
  if (entry) {
    ss << " " << std::hex << entry->addr;
  } else {
    ss << " nullptr";
  }
  LOG(INFO) << ss.str();
  if (entry == nullptr) {
    LOG(WARNING) << "Tensor Method Empty: tensor without memory";
  }
  return TensorPool::Get()->Insert(STensor(entry, size.vec(), dtype));
}

uint64_t EmptyStrided(at::IntArrayRef size, at::IntArrayRef stride,
                      DLDataType dtype) {
  auto storage_nbytes = ComputeStorageNbytes(size, stride, dtype);
  auto entry = CUDAMemPool::Get()->Alloc(storage_nbytes);
  std::stringstream ss;
  ss << "Create empty_strided size " << size << " stride " << stride << " nbytes " << storage_nbytes;
  if (entry) {
    ss << " " << std::hex << entry->addr;
  } else {
    ss << " nullptr";
  }
  LOG(INFO) << ss.str();
  if (entry == nullptr) {
    LOG(WARNING) << "Tensor Method EmptyStrided: tensor without memory";
  }
  return TensorPool::Get()->Insert(STensor(entry, size.vec(), stride.vec(), dtype, 0));
}

uint64_t ViewDtype(uint64_t handle, DLDataType dtype) {
  auto tensor = TensorPool::Get()->Tensor(handle);
  int64_t self_elem_size = tensor->dtype.bits >> 3;
  int64_t new_elem_size = dtype.bits >> 3;
  if (self_elem_size == new_elem_size) {
    return TensorPool::Get()->Insert(STensor(tensor.MData(), tensor.ShapeVec(), tensor.StrideVec(), dtype, tensor.StorageOffset()));
  } else if (tensor->ndim == 0) {
    CHECK(false) << "tensor " << handle << " 0 dim to view " << tensor->dtype << " to " << dtype;
  } else if (self_elem_size > new_elem_size) {
    // Downsizing element size
    int64_t size_ratio = self_elem_size / new_elem_size;
    auto new_stride = ComputeStridesForViewDtypeDownsize(
        tensor.Stride(), size_ratio, tensor->dtype, dtype);
    std::vector<int64_t> new_size(tensor->shape, tensor->shape + tensor->ndim);
    new_size[tensor->ndim - 1] *= size_ratio;
    auto new_storage_offset = size_ratio * tensor.StorageOffset();
    return TensorPool::Get()->Insert(STensor(tensor.MData(), std::move(new_size), std::move(new_stride), dtype, new_storage_offset));
  } else {
    // Upsizing element size
    int64_t size_ratio = new_elem_size / self_elem_size;
    CHECK_EQ(tensor->shape[tensor->ndim - 1] % size_ratio, 0)
        << "size[-1] must to be divisible to view " << tensor->dtype << " as " << dtype;
    CHECK_EQ(tensor.StorageOffset() % size_ratio, 0)
        << "storage_offset must to be divisible to view " << tensor->dtype << " as " << dtype;

    auto new_stride = ComputeStridesForViewDtypeDownsize(
        tensor.Stride(), size_ratio, tensor->dtype, dtype);
    std::vector<int64_t> new_size(tensor->shape, tensor->shape + tensor->ndim);
    new_size[tensor->ndim - 1] /= size_ratio;
    auto new_storage_offset = tensor.StorageOffset() / size_ratio;
    return TensorPool::Get()->Insert(STensor(tensor.MData(), std::move(new_size), std::move(new_stride), dtype, new_storage_offset));
  }
}

uint64_t AsStrided(uint64_t handle, at::IntArrayRef size,
                   at::IntArrayRef stride, c10::optional<int64_t> storage_offset) {
  std::cout << "Astrided " << size << " " << stride << " " <<  storage_offset.value_or(0) << std::endl;
  auto tensor = TensorPool::Get()->Tensor(handle);
  CheckMemoryBound(size, stride, tensor->dtype, storage_offset.value_or(0), tensor.MData());
  return TensorPool::Get()->Insert(STensor(tensor.MData(), size.vec(), stride.vec(), tensor->dtype, storage_offset.value_or(0)));
}

void STensor::Resize(at::IntArrayRef size, at::OptionalIntArrayRef stride) {
  // std::cout << "resize tensor " << size << std::endl;
  bool same_size = size.size() == get()->tensor_.ndim;
  if (same_size) {
    for (size_t i = 0; i < size.size(); i++) {
      if (size[i] != get()->tensor_.shape[i]) {
        same_size = false;
        break;
      }
    }
  }
  if (same_size) {
    return;
  }

  size_t storage_nbytes;
  if (stride.has_value()) {
    storage_nbytes = ComputeStorageNbytes(size, stride.value(), get()->tensor_.dtype) 
        + get()->tensor_.byte_offset;
  } else {
    storage_nbytes = ComputeStorageNbytes(size, get()->tensor_.dtype) 
        + get()->tensor_.byte_offset;
  }

  // std::cout << "storage_nbytes: " << storage_nbytes << std::endl;
  std::stringstream ss;
  ss << "Resize storage_nbytes: " << storage_nbytes;
  TensorContainer::memory_data_t mdata = MData();
  if (mdata == nullptr || mdata->nbytes < storage_nbytes) {
    auto new_mdata = CUDAMemPool::Get()->Resize(mdata, storage_nbytes);
    CUDAMemPool::Get()->CopyFromTo(mdata, new_mdata);
    mdata = new_mdata;
  }
  if (mdata) {
    ss << " mdata:" << mdata->addr << " " << mdata->nbytes;
  }
  std::cout << ss.str() << std::endl;

  if (stride.has_value()) {
    get()->SetTensor(mdata, size.vec(), stride.value().vec(), 
        get()->tensor_.dtype, std::nullopt);
  } else {
    get()->SetTensor(mdata, size.vec(), get()->tensor_.dtype, std::nullopt);
  }
}

}
}