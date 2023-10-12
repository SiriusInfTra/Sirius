#include <ATen/ATen.h>
#include <c10/core/TensorOptions.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <torch/library.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/NativeFunctions.h>
#include <ATen/InferSize.h>

#include <sta/tensor_pool.h>
#include <sta/tensor_methods.h>
#include "tensor_impl.h"
#include "dlpack_convert.h"
#include "convolution.h"

#include <glog/logging.h>


namespace torch_col {

namespace {
using namespace colserve;

inline ColTensorImpl* GetColTensorImpl(const at::Tensor& tensor) {
  auto impl = dynamic_cast<ColTensorImpl*>(tensor.unsafeGetTensorImpl());
  CHECK(impl) << "input tensor is not a ColTensor " << tensor.toString();
  return impl;
}

void cuda_fallback(const c10::OperatorHandle &op, torch::jit::Stack *stack) {
  auto schema = op.schema();
  std::cout << "redispatching " << schema << " to CUDA" << std::endl;
  op.redispatchBoxed(c10::DispatchKeySet(c10::DispatchKey::CUDA), stack);
}

at::Tensor empty(
    at::IntArrayRef size, c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout, c10::optional<at::Device> device, 
    c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  CHECK(!device.has_value() || device.value().is_cuda());
  auto scalar_type = at::dtype_or_default(dtype);
  auto dlpack_dtype = getDLDataType(scalar_type);
  // auto handle = sta::TensorPool::Get()->Empty(size_vec, dlpack_dtype);
  auto handle = sta::Empty(size.vec(), dlpack_dtype);
  return at::detail::make_tensor_base<ColTensorImpl>(std::make_shared<ColTensorImpl::Data>(handle));
}

at::Tensor empty_strided(
    at::IntArrayRef size, at::IntArrayRef stride, 
    c10::optional<at::ScalarType> dtype_opt, c10::optional<at::Layout> layout_opt, 
    c10::optional<at::Device> device_opt, c10::optional<bool> pin_memory_opt) {
  CHECK(!device_opt.has_value() || device_opt.value().is_cuda());
  CHECK(!pin_memory_opt.has_value() || !pin_memory_opt.value());
  DCHECK(!layout_opt.has_value() || layout_opt.value() == at::kStrided);

  auto scalar_type = at::dtype_or_default(dtype_opt);
  auto dlpack_dtype = getDLDataType(scalar_type);
  auto handle = sta::EmptyStrided(size.vec(), stride.vec(), dlpack_dtype);
  return MakeColTensor(handle);
} 

at::Tensor as_strided(
    const at::Tensor& self, at::IntArrayRef size, at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset) {
  auto impl = GetColTensorImpl(self);
  auto handle = sta::AsStrided(
      impl->Handle(), size, stride, storage_offset);
  // return at::detail::make_tensor_base<ColTensorImpl>(std::make_shared<ColTensorImpl::Data>(handle));
  return MakeColTensorAlias(handle, self);
}

at::Tensor _reshape_alias(const at::Tensor& self, at::IntArrayRef size, at::IntArrayRef stride) {
  auto impl = GetColTensorImpl(self);
  return MakeColTensorAlias(sta::AsStrided(
      impl->Handle(), size, stride, impl->storage_offset()), self);
}

const at::Tensor& resize_(const at::Tensor& self, at::IntArrayRef size, c10::optional<at::MemoryFormat> memory_format) {
  auto impl = GetColTensorImpl(self);
  impl->Tensor().Resize(size, c10::nullopt);
  std::cout << "resize_ " << size << " new ts " << self.sizes() << " " << self.numel() << " "
            << self.data_ptr() << std::endl;
  return self;
}

at::Tensor view(const at::Tensor& self, at::IntArrayRef size) {
  auto impl = GetColTensorImpl(self);
  at::DimVector inferred_size = at::infer_size_dv(size, self.numel());
  auto stride = at::detail::computeStride(self.sizes(),
                                          self.strides(),
                                          inferred_size);
  CHECK(stride.has_value()) << "view size is "
    "not compatible with input tensor's size and stride (at least one dimension"
    " spans across two contiguous subspaces). Use .reshape(...) instead.";
  return MakeColTensorAlias(sta::AsStrided(
      impl->Handle(), inferred_size, stride.value(), impl->storage_offset()), self);
}

at::Tensor view_dtype(const at::Tensor &self, at::ScalarType dtype) {
  if (self.dtype() == dtype) {
    return self;
  }
  auto impl = GetColTensorImpl(self);
  return MakeColTensorAlias(sta::ViewDtype(
      impl->Handle(), getDLDataType(dtype)), self);
}

at::Tensor alias(const at::Tensor &self) {
  auto impl = GetColTensorImpl(self);
  return MakeColTensorAlias(sta::AsStrided(
      impl->Handle(), self.sizes(), self.strides(), impl->storage_offset()), self);
}



TORCH_LIBRARY_IMPL(aten, CUDA, m) {
  m.impl("empty.memory_format", TORCH_FN(empty));
  m.impl("empty_strided", TORCH_FN(empty_strided));
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("as_strided", TORCH_FN(as_strided));
  m.impl("_reshape_alias", TORCH_FN(_reshape_alias));
  m.impl("resize_", TORCH_FN(resize_));
  m.impl("view", TORCH_FN(view));
  m.impl("view.dtype", TORCH_FN(view_dtype));
  m.impl("alias", TORCH_FN(alias));

  // m.impl("convolution_overrideable", convolution);
  m.impl("_convolution", TORCH_FN(_convolution));
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<
      &cuda_fallback>());
}

}
}