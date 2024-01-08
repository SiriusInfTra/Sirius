#include <c10/core/MemoryFormat.h>
#include <sta/tensor_methods.h>

#include "tensor_impl.h"
#include "dlpack_convert.h"

#include <glog/logging.h>

namespace torch_col {

using namespace colserve;

ColTensorImpl::ColTensorImpl(std::shared_ptr<Data> data)
    : c10::TensorImpl(c10::DispatchKeySet{c10::DispatchKey::PrivateUse1,
                                          c10::DispatchKey::AutogradPrivateUse1},
                      GetTypeMeta(data),
                      c10::Device{c10::DeviceType::CUDA, 0}),
      data_(data) {
  // LOG(INFO) << "ColTensorImpl" << std::endl;
  set_sizes_strides_policy(SizesStridesPolicy::CustomSizes);
  // UpdateStorage();
  auto tensor = sta::TensorPool::Get()->Tensor(data_->handle);
  auto mdata = tensor.MData();
  storage_ = at::Storage{{}, mdata ? mdata->nbytes : 0, 
      c10::DataPtr{mdata ? mdata->addr : nullptr, 
      c10::Device{c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(tensor->device.device_id)}}};
  // storage_offset_ = tensor->byte_offset / (tensor->dtype.bits >> 3);
  storage_offset_ = tensor->byte_offset / sta::GetDataTypeNbytes(tensor->dtype);
  // set_sizes_and_strides(tensor.Shape(), tensor.Stride());
  UpdateSize();
  UpdateVersion();
  refresh_contiguous();
}


ColTensorImpl::ColTensorImpl(std::shared_ptr<Data> data,
                             const at::Storage &storage)
    : c10::TensorImpl(c10::DispatchKeySet{c10::DispatchKey::PrivateUse1,
                                          c10::DispatchKey::AutogradPrivateUse1},
                      GetTypeMeta(data),
                      c10::Device{c10::DeviceType::CUDA, 0}),
      data_(data) {
  auto tensor = sta::TensorPool::Get()->Tensor(data_->handle);
  // LOG(INFO) << "ColTensorImpl w/ storage" << std::endl;
  set_sizes_strides_policy(SizesStridesPolicy::CustomSizes);
  storage_ = storage;
  // storage_offset_ = tensor->byte_offset / (tensor->dtype.bits >> 3);
  storage_offset_ = tensor->byte_offset / sta::GetDataTypeNbytes(tensor->dtype);
  // UpdateSize();
  UpdateSize();
  UpdateVersion();
  refresh_contiguous();
  // set_sizes_and_strides(tensor.Shape(), tensor.Stride());
}

sta::STensor ColTensorImpl::Tensor() const {
  return sta::TensorPool::Get()->Tensor(data_->handle);
}

const sta::STensor ColTensorImpl::CTensor() const {
  return sta::TensorPool::Get()->CTensor(data_->handle);
}

at::IntArrayRef ColTensorImpl::sizes_custom() const {
  // auto tensor = sta::TensorPool::Get()->Tensor(data_->handle);
  // return at::IntArrayRef(tensor->shape, tensor->ndim);
  const_cast<ColTensorImpl*>(this)->UpdateAll();
  return sizes_default();
}

at::IntArrayRef ColTensorImpl::strides_custom() const {
  // auto tensor = colserve::sta::TensorPool::Get()->Tensor(data_->handle);
  // return at::IntArrayRef(tensor->strides, tensor->ndim);
  const_cast<ColTensorImpl*>(this)->UpdateAll();
  return strides_default();
}

int64_t ColTensorImpl::dim_custom() const {
  // auto tensor = colserve::sta::TensorPool::Get()->Tensor(data_->handle);
  // return tensor->ndim;
  const_cast<ColTensorImpl*>(this)->UpdateAll();
  return dim_default();
}

int64_t ColTensorImpl::numel_custom() const {
  // int64_t numel = 1;
  // for (auto dim : sizes_custom()) {
  //   numel *= dim;
  // }
  // return numel;
  const_cast<ColTensorImpl*>(this)->UpdateAll();
  return numel_default();
}

bool ColTensorImpl::is_contiguous_custom(at::MemoryFormat memory_format) const {
  // const_cast<ColTensorImpl*>(this)->is_contiguous_ = Tensor().ComputeContiguous();
  const_cast<ColTensorImpl*>(this)->UpdateAll();
  return is_contiguous_default(memory_format);
}

bool ColTensorImpl::has_storage() const {
  const_cast<ColTensorImpl*>(this)->UpdateAll();
  return storage_;
}

const at::Storage& ColTensorImpl::storage() const {
  const_cast<ColTensorImpl*>(this)->UpdateAll();
  return storage_;
}

int64_t ColTensorImpl::storage_offset() const {
  // auto tensor = Tensor();
  // return tensor.StorageOffset();
  const_cast<ColTensorImpl*>(this)->UpdateAll();
  return storage_offset_;
}

c10::intrusive_ptr<c10::TensorImpl> ColTensorImpl::shallow_copy_and_detach(
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const {
  return shallow_copy_and_detach_core_custom(
      version_counter, allow_tensor_metadata_change);
}

c10::intrusive_ptr<c10::TensorImpl> ColTensorImpl::shallow_copy_and_detach(
    c10::VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const {
  return shallow_copy_and_detach_core_custom(
      std::move(version_counter), allow_tensor_metadata_change);
}

template <typename VariableVersion>
c10::intrusive_ptr<c10::TensorImpl> ColTensorImpl::shallow_copy_and_detach_core_custom(
    VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const {
  auto impl = c10::make_intrusive<ColTensorImpl>(data_, storage_);
  copy_tensor_metadata(
      /*src_impl=*/this,
      /*dest_impl=*/impl.get(),
      /*version_counter=*/std::forward<VariableVersion>(version_counter),
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
  return impl;
}

caffe2::TypeMeta ColTensorImpl::GetTypeMeta(const std::shared_ptr<Data> &data) {
  auto tensor = sta::TensorPool::Get()->Tensor(data->handle);
  return getCaffeTypeMeta(tensor->dtype);
}

void ColTensorImpl::UpdateStorage() {
  auto tensor = sta::TensorPool::Get()->Tensor(data_->handle);
  auto mdata = tensor.MData();
  
  if (mdata) {
    storage_.set_data_ptr_noswap(c10::DataPtr{
        mdata->addr, c10::Device{c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(tensor->device.device_id)}});
    storage_.set_nbytes(mdata->nbytes);
  } else {
    storage_.set_data_ptr_noswap(c10::DataPtr{
        nullptr, c10::Device{c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(tensor->device.device_id)}});
    storage_.set_nbytes(0);
  }
  // storage_offset_ = tensor->byte_offset / (tensor->dtype.bits >> 3);
  storage_offset_ = tensor->byte_offset / sta::GetDataTypeNbytes(tensor->dtype);

  if (storage_.data() != nullptr) {
    CHECK(sta::CUDAMemPool::Get()->CheckAddr(storage_.data()));
    CHECK(sta::CUDAMemPool::Get()->CheckAddr(static_cast<char*>(storage_.data()) + storage_.nbytes()));
  }

  DCHECK(!tensor.ComputeContiguous() || tensor.ComputeNumel() * (sta::GetDataTypeNbytes(tensor->dtype)) <= storage_.nbytes())
    << "numel: " << tensor.ComputeNumel() << " dtype: " << sta::GetDataTypeNbytes(tensor->dtype) 
    << " storage: " << storage_.nbytes() << " size " << tensor.Shape() 
    << " handle " << data_->handle << " mdata->nbytes " << mdata->nbytes;

  // }
  // if (mdata != nullptr)
  //   std::cout << "mdata: " << std::hex << mdata->addr << " " << mdata->size << " "
  //             << static_cast<void*>(static_cast<char*>(storage_.data()) + tensor->byte_offset) 
  //             << std::endl;
}

void ColTensorImpl::UpdateSize() {
  auto tensor = sta::TensorPool::Get()->Tensor(data_->handle);
  // set_sizes_and_strides(tensor.Shape(), tensor.Stride());

  const auto new_dim = tensor->ndim;
  
  auto new_size = tensor.Shape();
  auto new_stride = tensor.Stride();

  sizes_and_strides_.set_sizes(new_size);

  if (new_dim > 0) {
    for (size_t dim = new_dim - 1;; dim--) {
      if (new_stride[dim] >= 0) {
        sizes_and_strides_.stride_at_unchecked(dim) = new_stride[dim];
      } else {
        LOG(FATAL) << "STensor got negtive stride, " << new_stride[dim] 
                   << " at dim " << dim;
        // // XXX: This behavior is surprising and may need to be removed to
        // // support negative strides. Some pytorch functions rely on it:
        // // for example, torch.cat (run TestTorch.test_cat_empty).
        // if (dim == new_dim - 1) {
        //   sizes_and_strides_.stride_at_unchecked(dim) = 1;
        // } else {
        //   // Keep stride monotonically increasing to match NumPy.
        //   sizes_and_strides_.stride_at_unchecked(dim) =
        //       std::max<int64_t>(
        //           sizes_and_strides_.size_at_unchecked(dim + 1), 1) *
        //       sizes_and_strides_.stride_at_unchecked(dim + 1);
        // }
      }
      if (dim == 0)
        break;
    }
  }

  // is_contiguous_ = tensor.ComputeContiguous();
  numel_ = tensor.ComputeNumel();
}

void ColTensorImpl::UpdateAll() {
  if (!IsUpdated()) {
    UpdateStorage();
    UpdateSize();
    UpdateVersion();
    refresh_contiguous();
  }
}

at::Tensor MakeColTensorEmpty(at::IntArrayRef size, const at::TensorOptions &options) {
  CHECK(!options.has_device() || options.device_opt().value().is_cuda());
  CHECK(!options.has_memory_format() || options.memory_format_opt().value() == at::MemoryFormat::Contiguous);
  auto scalar_type = at::dtype_or_default(options.dtype_opt());
  auto dlpack_dtype = getDLDataType(scalar_type);
  // auto handle = sta::TensorPool::Get()->Empty(size_vec, dlpack_dtype);
  auto handle = colserve::sta::Empty(size, at::MemoryFormat::Contiguous, dlpack_dtype, sta::MemType::kTrain);
  return MakeColTensor(handle);
}

at::Tensor MakeColTensorEmpty(at::IntArrayRef size, at::ScalarType scalar_type) {
  auto dlpack_dtype = getDLDataType(scalar_type);
  auto handle = colserve::sta::Empty(size, at::MemoryFormat::Contiguous, dlpack_dtype, sta::MemType::kTrain);
  return MakeColTensor(handle);

}


}