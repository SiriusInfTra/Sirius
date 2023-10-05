#include "tensor_impl.h"
#include "dlpack_convert.h"

#include <undef_log.h>
#include <glog/logging.h>


namespace torch_col {
  
ColTensorImpl::ColTensorImpl(std::shared_ptr<Data> data)
    : c10::TensorImpl(c10::DispatchKeySet{c10::DispatchKey::PrivateUse1,
                                          c10::DispatchKey::AutogradPrivateUse1},
                      GetTypeMeta(data),
                      c10::DeviceType::CUDA),
      data_(data) {
  LOG(INFO) << "ColTensorImpl" << std::endl;
  set_sizes_strides_policy(SizesStridesPolicy::CustomSizes);
}

colserve::sta::STensor ColTensorImpl::Tensor() const {
  return colserve::sta::TensorPool::Get()->Tensor(data_->handle);
}

const colserve::sta::STensor ColTensorImpl::CTensor() const {
  return colserve::sta::TensorPool::Get()->CTensor(data_->handle);
}

caffe2::TypeMeta ColTensorImpl::GetTypeMeta(const std::shared_ptr<Data> &data) {
  auto tensor = colserve::sta::TensorPool::Get()->Tensor(data->handle);
  return getCaffeTypeMeta(tensor->dtype);
}

at::IntArrayRef ColTensorImpl::sizes_custom() const {
  auto tensor = colserve::sta::TensorPool::Get()->Tensor(data_->handle);
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
  return true;
}

}