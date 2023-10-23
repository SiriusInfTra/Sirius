#include <sta/tensor_methods.h>

#include "tensor_impl.h"
#include "dlpack_convert.h"

#include <undef_log.h>
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
  storage_offset_ = tensor->byte_offset / (tensor->dtype.bits >> 3);
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
  storage_offset_ = tensor->byte_offset / (tensor->dtype.bits >> 3);
}

sta::STensor ColTensorImpl::Tensor() const {
  return sta::TensorPool::Get()->Tensor(data_->handle);
}

const sta::STensor ColTensorImpl::CTensor() const {
  return sta::TensorPool::Get()->CTensor(data_->handle);
}

at::IntArrayRef ColTensorImpl::sizes_custom() const {
  auto tensor = sta::TensorPool::Get()->Tensor(data_->handle);
  return at::IntArrayRef(tensor->shape, tensor->ndim);
}

at::IntArrayRef ColTensorImpl::strides_custom() const {
  auto tensor = colserve::sta::TensorPool::Get()->Tensor(data_->handle);
  return at::IntArrayRef(tensor->strides, tensor->ndim);
}

int64_t ColTensorImpl::dim_custom() const {
  auto tensor = colserve::sta::TensorPool::Get()->Tensor(data_->handle);
  return tensor->ndim;
}

int64_t ColTensorImpl::numel_custom() const {
  auto numel = 1;
  for (auto dim : sizes_custom()) {
    numel *= dim;
  }
  return numel;
}

bool ColTensorImpl::is_contiguous_custom(at::MemoryFormat memory_format) const {
  return Tensor().ComputeContiguous();
}

bool ColTensorImpl::has_storage() const {
  const_cast<ColTensorImpl*>(this)->UpdateStorage();
  return storage_;
}

const at::Storage& ColTensorImpl::storage() const {
  const_cast<ColTensorImpl*>(this)->UpdateStorage();
  return storage_;
}

int64_t ColTensorImpl::storage_offset() const {
  auto tensor = Tensor();
  return tensor.StorageOffset();
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
  }
  storage_offset_ = tensor->byte_offset / (tensor->dtype.bits >> 3);

  // if (mdata != nullptr)
  //   std::cout << "mdata: " << std::hex << mdata->addr << " " << mdata->size << " "
  //             << static_cast<void*>(static_cast<char*>(storage_.data()) + tensor->byte_offset) 
  //             << std::endl;
}

at::Tensor MakeColTensorEmpty(at::IntArrayRef size, const at::TensorOptions &options) {
  CHECK(!options.has_device() || options.device_opt().value().is_cuda());
  CHECK(!options.has_memory_format() || options.memory_format_opt().value() == at::MemoryFormat::Contiguous);
  auto scalar_type = at::dtype_or_default(options.dtype_opt());
  auto dlpack_dtype = getDLDataType(scalar_type);
  // auto handle = sta::TensorPool::Get()->Empty(size_vec, dlpack_dtype);
  auto handle = colserve::sta::Empty(size, dlpack_dtype, sta::MemType::kTrain);
  return MakeColTensor(handle);
}

at::Tensor MakeColTensorEmpty(at::IntArrayRef size, at::ScalarType scalar_type) {
  auto dlpack_dtype = getDLDataType(scalar_type);
  auto handle = colserve::sta::Empty(size, dlpack_dtype, sta::MemType::kTrain);
  return MakeColTensor(handle);

}


}