#include <ATen/native/TensorConversions.h>
#include <cuda_runtime_api.h>

#include "tensor_methods.h"
#include "tensor_pool.h"
#include "shape_helper.h"
#include "dtype_helper.h"

namespace colserve {
namespace sta {

uint64_t Null(at::IntArrayRef size, DLDataType dtype) {
  return TensorPool::Get()->Insert(STensor(size.vec(), dtype));
}

STensor RawNull(at::IntArrayRef size, DLDataType dtype) {
  return STensor(size.vec(), dtype);
}

uint64_t Empty(at::IntArrayRef size, DLDataType dtype, MemType mtype) {
  auto storage_nbytes = ComputeStorageNbytes(size, dtype);
  auto entry = CUDAMemPool::Get()->Alloc(storage_nbytes, mtype);
  // std::stringstream ss;
  // ss << "Create empty size " << size << " nbytes " << storage_nbytes;
  // if (entry) {
  //   ss << " " << std::hex << entry->addr;
  // } else {
  //   ss << " nullptr";
  // }
  // LOG(INFO) << ss.str();
  if (entry == nullptr) {
    DLOG(WARNING) << "Tensor Method Empty: tensor without memory";
  }
  return TensorPool::Get()->Insert(STensor(entry, size.vec(), dtype));
}

STensor RawEmpty(at::IntArrayRef size, DLDataType dtype, MemType mtype) {
  auto storage_nbytes = ComputeStorageNbytes(size, dtype);
  auto entry = CUDAMemPool::RawAlloc(storage_nbytes, mtype);
  return STensor(entry, size.vec(), dtype);
}

uint64_t EmptyStrided(at::IntArrayRef size, at::IntArrayRef stride,
                      DLDataType dtype, MemType mtype) {
  auto storage_nbytes = ComputeStorageNbytes(size, stride, dtype);
  auto entry = CUDAMemPool::Get()->Alloc(storage_nbytes, mtype);
  // std::stringstream ss;
  // ss << "Create empty_strided size " << size << " stride " << stride << " nbytes " << storage_nbytes;
  // if (entry) {
  //   ss << " " << std::hex << entry->addr;
  // } else {
  //   ss << " nullptr";
  // }
  // LOG(INFO) << ss.str();
  if (entry == nullptr) {
    DLOG(WARNING) << "Tensor Method EmptyStrided: tensor without memory";
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

uint64_t ViewShapeDtype(uint64_t handle, at::IntArrayRef size, DLDataType dtype) {
  auto tensor = TensorPool::Get()->Tensor(handle);
  // auto new_stride = ComputeStrides(size);
  // int64_t new_elem_size = dtype.bits >> 3;
  // auto bytes_offset = tensor->byte_offset;
  // CHECK_EQ(bytes_offset % new_elem_size, 0);
  // auto new_storage_offset = bytes_offset / new_elem_size;
  // if (!tensor.IsNull()) {
  //   CheckMemoryBound(tensor.Shape(), new_stride, dtype, new_storage_offset, tensor.MData());
  //   return TensorPool::Get()->Insert(STensor(tensor.MData(), size.vec(), std::move(new_stride), dtype, new_storage_offset));
  // } else {
  //   return TensorPool::Get()->Insert(STensor(size.vec(), std::move(new_stride), dtype, new_storage_offset));
  // }
  auto view_tensor = RawViewShapeDtype(tensor, size, dtype);
  return TensorPool::Get()->Insert(view_tensor);
}

STensor RawViewShapeDtype(STensor tensor, at::IntArrayRef size, DLDataType dtype) {
  auto new_stride = ComputeStrides(size);
  int64_t new_elem_size = dtype.bits >> 3;
  auto bytes_offset = tensor->byte_offset;
  CHECK_EQ(bytes_offset % new_elem_size, 0);
  auto new_storage_offset = bytes_offset / new_elem_size;
  if (!tensor.IsNull()) {
    CheckMemoryBound(tensor.Shape(), new_stride, dtype, new_storage_offset, tensor.MData());
    return STensor(tensor.MData(), size.vec(), std::move(new_stride), dtype, new_storage_offset);
  } else {
    return STensor(size.vec(), std::move(new_stride), dtype, new_storage_offset);
  }
}

uint64_t AsStrided(uint64_t handle, at::IntArrayRef size,
                   at::IntArrayRef stride, c10::optional<int64_t> storage_offset) {
  // DLOG(INFO) << "Astrided " << size << " " << stride << " " <<  storage_offset.value_or(0) << std::endl;
  auto tensor = TensorPool::Get()->Tensor(handle);
  if (!tensor.IsNull()) {
    CheckMemoryBound(size, stride, tensor->dtype, storage_offset.value_or(0), tensor.MData());
    return TensorPool::Get()->Insert(STensor(tensor.MData(), size.vec(), stride.vec(), tensor->dtype, storage_offset.value_or(0)));
  } else {
    return TensorPool::Get()->Insert(STensor(size.vec(), stride.vec(), tensor->dtype, storage_offset.value_or(0)));
  }
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

  std::stringstream ss;
  // ss << "Resize storage_nbytes: " << storage_nbytes;
  TensorContainer::memory_data_t mdata = MData();
  if (mdata == nullptr || mdata->nbytes < storage_nbytes) {
    auto new_mdata = CUDAMemPool::Get()->Resize(mdata, storage_nbytes);
    // CUDAMemPool::Get()->CopyFromTo(mdata, new_mdata);
    mdata = new_mdata;
  }
  // if (mdata) {
  //   ss << " mdata:" << mdata->addr << " " << mdata->nbytes;
  // }
  // std::cout << ss.str() << std::endl;
  DLOG(INFO) << "Resize storage_nbytes: " << storage_nbytes << " addr " << (mdata ? mdata->addr : 0);

  if (stride.has_value()) {
    get()->SetTensor(mdata, size.vec(), stride.value().vec(), 
        get()->tensor_.dtype, std::nullopt);
  } else {
    get()->SetTensor(mdata, size.vec(), get()->tensor_.dtype, std::nullopt);
  }
}

void STensor::AllocForNull(MemType mtype, bool raw_alloc) {
  CHECK(IsNull());
  auto storage_nbytes = ComputeStorageNbytes(
      get()->shape_, get()->stride_, get()->tensor_.dtype, StorageOffset());
  TensorContainer::memory_data_t mdata;
  if (!raw_alloc) {
    mdata = CUDAMemPool::Get()->Alloc(storage_nbytes, mtype);
  } else {
    mdata = CUDAMemPool::RawAlloc(storage_nbytes, mtype);
  }
  if (storage_nbytes > 0 && mdata == nullptr) {
    LOG(FATAL) << "Tensor AllocForNull: tensor without memory";
  }
  get()->mdata_ = mdata;
  get()->tensor_.data = mdata->addr;
  get()->is_null_ = false;
}

void STensor::AssignMDataForNull(TensorContainer::memory_data_t mdata, bool check_memory_bound) {
  CHECK(IsNull());
  if (check_memory_bound) {
    CheckMemoryBound(get()->shape_, get()->stride_, get()->tensor_.dtype, StorageOffset(), mdata);
  }
  CHECK_NE(mdata, nullptr);
  get()->mdata_ = mdata;
  get()->tensor_.data = mdata->addr;
  get()->is_null_ = false;
}

void STensor::DeallocToNull() {
  CHECK(!IsNull());
  get()->tensor_.data = 0;
  get()->mdata_ = nullptr;
  get()->is_null_ = true;
}

}
}