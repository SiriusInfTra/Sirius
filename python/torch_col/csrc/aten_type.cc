#include <ATen/Tensor.h>
#include <c10/core/TensorOptions.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <torch/library.h>

#include <sta/tensor_pool.h>
#include <sta/tensor_methods.h>
#include "tensor_impl.h"
#include "dlpack_convert.h"


namespace torch_col {

namespace {
at::Tensor empty(
    at::IntArrayRef size, c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout, c10::optional<at::Device> device, 
    c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  CHECK(!device.has_value() || device.value().is_cuda());
  auto scalar_type = at::dtype_or_default(dtype);
  auto dlpack_dtype = getDLDataType(scalar_type);
  std::vector<int64_t> size_vec(size.begin(), size.end());
  // auto handle = colserve::sta::TensorPool::Get()->Empty(size_vec, dlpack_dtype);
  auto handle = colserve::sta::Empty(size_vec, dlpack_dtype);
  return at::detail::make_tensor_base<ColTensorImpl>(std::make_shared<ColTensorImpl::Data>(handle));
}

at::Tensor as_strided(
    const at::Tensor& self, at::IntArrayRef size, at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset) {
  auto impl = dynamic_cast<ColTensorImpl*>(self.unsafeGetTensorImpl());
  CHECK(impl) << "input tensor is not a ColTensor " << self.toString();
  std::vector<int64_t> size_vec(size.begin(), size.end());
  std::vector<int64_t> stride_vec(stride.begin(), stride.end());
  auto handle = colserve::sta::AsStrided(
      impl->Handle(), size_vec, stride_vec, storage_offset.value_or(0));
  return at::detail::make_tensor_base<ColTensorImpl>(std::make_shared<ColTensorImpl::Data>(handle));
}
}


TORCH_LIBRARY_IMPL(aten, CUDA, m) {
  m.impl("empty.memory_format", TORCH_FN(empty));
  m.impl("as_strided", TORCH_FN(as_strided));
}


}