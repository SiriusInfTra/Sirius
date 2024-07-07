#include "log_as_glog_sta.h"
#include <ATen/core/ATen_fwd.h>
#include <ATen/native/TensorConversions.h>
#include <cuda_runtime_api.h>

#include "init.h"
#include "tensor_methods.h"
#include "tensor.h"
#include "shape_helper.h"
#include "dtype_helper.h"

namespace colserve {
namespace sta {

STensor Null(at::IntArrayRef size, DLDevice device, DLDataType dtype) {
  return STensor(size.vec(), device, dtype);
}

STensor Empty(at::IntArrayRef size, at::MemoryFormat memory_format, 
              DLDevice device, DLDataType dtype, MemType mtype) {
  CHECK(device.device_type == kDLCUDA);
  auto storage_nbytes = ComputeStorageNbytes(size, dtype);
  std::shared_ptr<CUDAMemPool::PoolEntry> entry;
  if (allocate_tensor_from_memory_pool) {
    entry = CUDAMemPool::Get()->Alloc(device.device_id, storage_nbytes, mtype, false);
  } else {
    entry = CUDAMemPool::RawAlloc(device.device_id, storage_nbytes, mtype);
  }
  CHECK(entry != nullptr && entry->nbytes >= storage_nbytes);
  return STensor(entry, size.vec(), memory_format, device, dtype);
}

// STensor RawEmpty(at::IntArrayRef size, DLDataType dtype, MemType mtype) {
//   auto storage_nbytes = ComputeStorageNbytes(size, dtype);
//   auto entry = CUDAMemPool::RawAlloc(storage_nbytes, mtype);
//   return STensor(entry, size.vec(), dtype);
// }

STensor EmptyStrided(at::IntArrayRef size, at::IntArrayRef stride,
                      DLDevice device, DLDataType dtype, MemType mtype) {
  CHECK(device.device_type == kDLCUDA);
  auto storage_nbytes = ComputeStorageNbytes(size, stride, dtype);
  std::shared_ptr<CUDAMemPool::PoolEntry> entry;
  if (allocate_tensor_from_memory_pool) {
    entry = CUDAMemPool::Get()->Alloc(device.device_id, storage_nbytes, mtype, false);
  } else {
    entry = CUDAMemPool::Get()->RawAlloc(device.device_id, storage_nbytes, mtype);
  }
  CHECK(entry != nullptr && entry->nbytes >= storage_nbytes);
  return STensor(entry, size.vec(), stride.vec(), device, dtype, 0);
}

STensor ViewDtype(const STensor tensor, DLDataType dtype) {
  // int64_t self_elem_size = tensor->dtype.bits >> 3;
  // int64_t new_elem_size = dtype.bits >> 3;
  int64_t self_elem_size = GetDataTypeNbytes(tensor->dtype);
  int64_t new_elem_size = GetDataTypeNbytes(dtype);
  CHECK(!tensor.IsNull());
  if (self_elem_size == new_elem_size) {
    return STensor(tensor.MData(), tensor.ShapeVec(), tensor.StrideVec(), 
                   tensor->device, dtype, tensor.StorageOffset());
  } else if (tensor->ndim == 0) {
    LOG(FATAL) << "tensor " << tensor << " 0 dim to view " << tensor->dtype << " to " << dtype;
  } else if (self_elem_size > new_elem_size) {
    // Downsizing element size
    int64_t size_ratio = self_elem_size / new_elem_size;
    auto new_stride = ComputeStridesForViewDtypeDownsize(
        tensor.Stride(), size_ratio, tensor->dtype, dtype);
    std::vector<int64_t> new_size(tensor->shape, tensor->shape + tensor->ndim);
    new_size[tensor->ndim - 1] *= size_ratio;
    auto new_storage_offset = size_ratio * tensor.StorageOffset();
    return STensor(tensor.MData(), std::move(new_size), std::move(new_stride), 
                   tensor->device, dtype, new_storage_offset);
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
    return STensor(tensor.MData(), std::move(new_size), std::move(new_stride), 
                   tensor->device, dtype, new_storage_offset);
  }
}

STensor ViewShapeDtype(const STensor tensor, at::IntArrayRef size, DLDataType dtype) {
  auto new_stride = ComputeStrides(size);
  // int64_t new_elem_size = dtype.bits >> 3;
  int64_t new_elem_size = sta::GetDataTypeNbytes(dtype);
  auto bytes_offset = tensor->byte_offset;
  CHECK_EQ(bytes_offset % new_elem_size, 0);
  auto new_storage_offset = bytes_offset / new_elem_size;
  if (!tensor.IsNull()) {
    CheckMemoryBound(tensor.Shape(), new_stride, dtype, new_storage_offset, tensor.MData());
    return STensor(tensor.MData(), size.vec(), std::move(new_stride), 
                   tensor->device, dtype, new_storage_offset);
  } else {
    return STensor(size.vec(), std::move(new_stride), 
                   tensor->device, dtype, new_storage_offset);
  }
}

STensor AsStrided(const STensor tensor, at::IntArrayRef size,
                   at::IntArrayRef stride, c10::optional<int64_t> storage_offset) {
  DLOG(INFO) << "sta::AsStrided"
             << " size " << size << " stride " << stride 
             << " storage_offset " <<  storage_offset.value_or(tensor.StorageOffset());
  if (!tensor.IsNull()) {
    CheckMemoryBound(size, stride, tensor->dtype, 
        storage_offset.value_or(tensor.StorageOffset()), tensor.MData());
    return STensor(tensor.MData(), size.vec(), stride.vec(), 
                   tensor->device, tensor->dtype, 
                   storage_offset.value_or(tensor.StorageOffset()));
  } else {
    return STensor(size.vec(), stride.vec(), 
                   tensor->device, tensor->dtype, 
                   storage_offset.value_or(tensor.StorageOffset()));
  }
}

void AsStrided_(STensor tensor, at::IntArrayRef size,
                at::IntArrayRef stride, c10::optional<int64_t> storage_offset) {
  DLOG(INFO) << "sta::AsStrided_"
             << " size " << size << " stride " << stride 
             << " storage_offset " <<  storage_offset.value_or(tensor.StorageOffset());
  if (!tensor.IsNull()) {
    CheckMemoryBound(size, stride, tensor->dtype, 
        storage_offset.value_or(tensor.StorageOffset()), tensor.MData());
    tensor.get()->SetTensor(tensor.MData(), size.vec(), stride.vec(), tensor->device, tensor->dtype, 
        storage_offset.value_or(tensor.StorageOffset()));
  } else {
    tensor.get()->SetTensor(nullptr, size.vec(), stride.vec(), tensor->device, tensor->dtype, 
        storage_offset.value_or(tensor.StorageOffset()));
  }
  tensor.UpdateVersion();
}

// void STensor::Resize(at::IntArrayRef size, at::OptionalIntArrayRef stride) {
//   // std::cout << "resize tensor " << size << std::endl;
//   CHECK(!IsNull());
//   bool same_size = (Shape() == size) && (!stride.has_value() || Shape() == stride.value());
//   if (same_size) {
//     return;
//   }

//   size_t storage_nbytes;
//   if (stride.has_value()) {
//     storage_nbytes = ComputeStorageNbytes(size, stride.value(), get()->tensor_.dtype) 
//         + get()->tensor_.byte_offset;
//   } else {
//     storage_nbytes = ComputeStorageNbytes(size, get()->tensor_.dtype) 
//         + get()->tensor_.byte_offset;
//   }

//   // std::stringstream ss;
//   // ss << "Resize storage_nbytes: " << storage_nbytes;
//   TensorContainer::memory_data_t mdata = MData();
//   if (mdata->addr == nullptr || mdata->nbytes < storage_nbytes) {
//     auto new_mdata = CUDAMemPool::Get()->Resize(mdata, storage_nbytes);
//     CUDAMemPool::Get()->CopyFromTo(mdata, new_mdata);
//     mdata = new_mdata;
//   }
//   // if (mdata) {
//   //   ss << " mdata:" << mdata->addr << " " << mdata->nbytes;
//   // }
//   // std::cout << ss.str() << std::endl;
//   DLOG(INFO) << "Resize storage_nbytes: " << storage_nbytes << " addr " << (mdata ? mdata->addr : 0);

//   if (stride.has_value()) {
//     get()->SetTensor(mdata, size.vec(), stride.value().vec(), 
//         get()->tensor_.dtype, std::nullopt);
//   } else {
//     get()->SetTensor(mdata, size.vec(), get()->tensor_.dtype, std::nullopt);
//   }
//   this->UpdateVersion();
// }

void STensor::AllocForNull(MemType mtype) {
  CHECK(IsNull());
  auto storage_nbytes = ComputeStorageNbytes(
      get()->shape_, get()->stride_, get()->tensor_.dtype, StorageOffset());
  TensorContainer::memory_data_t mdata;
  if (allocate_tensor_from_memory_pool) {
    mdata = CUDAMemPool::Get()->Alloc(get()->tensor_.device.device_id, 
                                      storage_nbytes, mtype, false);
  } else {
    mdata = CUDAMemPool::RawAlloc(get()->tensor_.device.device_id, 
                                  storage_nbytes, mtype);
  }
  if (storage_nbytes > 0 && mdata == nullptr) {
    LOG(FATAL) << "Tensor AllocForNull: tensor without memory";
  }
  get()->mdata_ = mdata;
  get()->tensor_.data = mdata->addr;
  get()->is_null_ = false;
  this->UpdateVersion();
}

void STensor::SetMDataForNull(TensorContainer::memory_data_t mdata, bool check_memory_bound) {
  CHECK(IsNull());
  if (check_memory_bound) {
    CheckMemoryBound(get()->shape_, get()->stride_, get()->tensor_.dtype, StorageOffset(), mdata);
  }
  CHECK(mdata != nullptr);
  get()->mdata_ = mdata;
  get()->tensor_.data = mdata->addr;
  get()->is_null_ = false;
  this->UpdateVersion();
}

void STensor::DeallocToNull() {
  CHECK(!IsNull());
  get()->tensor_.data = 0;
  get()->mdata_ = nullptr;
  get()->is_null_ = true;
  this->UpdateVersion();
}

void STensor::DeallocToDummy() {
  if (get()->tensor_.data == (void*)-1) {
    return;
  }
  get()->tensor_.data = (void*)-1;
  auto nbytes = get()->mdata_->nbytes;
  auto mtype = get()->mdata_->mtype;
  get()->mdata_ = std::shared_ptr<CUDAMemPool::PoolEntry>(new CUDAMemPool::PoolEntry{(void*)-1, nbytes, mtype});
  this->UpdateVersion();
}

void STensor::Rearrange() {
  LOG(FATAL) << "deprecated Rearrange()";

  // if (IsNull() || (get()->tensor_.data == (void*)-1)) {
  //   return; // null/dummy tensor
  // }
  // auto new_mdata = CUDAMemPool::Get()->Alloc(
  //   get()->mdata_->nbytes, get()->mdata_->mtype, false);
  // // CUDAMemPool::Get()->CopyFromTo(get()->mdata_, new_mdata);
  // get()->mdata_ = new_mdata;
  // get()->tensor_.data = new_mdata->addr;
  // this->UpdateVersion();
}

}
}