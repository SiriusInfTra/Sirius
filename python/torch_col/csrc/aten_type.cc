#include <ATen/Tensor.h>
#include <c10/core/TensorOptions.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <torch/library.h>

#include <sta/tensor_pool.h>
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
  auto handle = colserve::sta::TensorPool::Get()->Empty(size_vec, dlpack_dtype);
  return at::detail::make_tensor_base<ColTensorImpl>(std::make_shared<ColTensorImpl::Data>(handle));
}
}


TORCH_LIBRARY_IMPL(aten, CUDA, m) {
  m.impl("empty.memory_format", TORCH_FN(empty));
}


}