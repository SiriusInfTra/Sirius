#include <ATen/Tensor.h>
#include <c10/core/TensorOptions.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <torch/library.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/Functions.h>

#include <sta/tensor_pool.h>
#include <sta/tensor_methods.h>
#include "tensor_impl.h"
#include "dlpack_convert.h"

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
  std::cout << "empty " << size << std::endl;
  auto scalar_type = at::dtype_or_default(dtype);
  auto dlpack_dtype = getDLDataType(scalar_type);
  // auto handle = sta::TensorPool::Get()->Empty(size_vec, dlpack_dtype);
  auto handle = sta::Empty(size.vec(), dlpack_dtype);
  return at::detail::make_tensor_base<ColTensorImpl>(std::make_shared<ColTensorImpl::Data>(handle));
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
  // std::cout << "new ts " << self.sizes() << " " << self.numel() << " "
  //           << self.data_ptr() << std::endl;
  return self;
}

}


TORCH_LIBRARY_IMPL(aten, CUDA, m) {
  m.impl("empty.memory_format", TORCH_FN(empty));
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("as_strided", TORCH_FN(as_strided));
  m.impl("_reshape_alias", TORCH_FN(_reshape_alias));
  m.impl("resize_", TORCH_FN(resize_));
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<
      &cuda_fallback>());
}

}