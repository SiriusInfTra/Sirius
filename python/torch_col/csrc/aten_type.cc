#include <ATen/ATen.h>
#include <c10/core/TensorOptions.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <torch/library.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/NativeFunctions.h>
#include <ATen/InferSize.h>
#include <ATen/core/op_registration/adaption.h>

#include <sta/init.h>
#include <sta/tensor_pool.h>
#include <sta/tensor_methods.h>
#include <sta/shape_helper.h>
#include "tensor_impl.h"
#include "dlpack_convert.h"
#include "convolution.h"
#include "cudnn/cudnn_custom.h"
#include "override_ops/override_ops.h"

#include <glog/logging.h>


namespace torch_col {

namespace {
using namespace colserve;

inline ColTensorImpl* GetColTensorImpl(const at::Tensor& tensor) {
  auto impl = dynamic_cast<ColTensorImpl*>(tensor.unsafeGetTensorImpl());
  CHECK(impl) << "input tensor is not a ColTensor, got *impl " << tensor.unsafeGetTensorImpl() 
              << " data_ptr " << tensor.data_ptr() << " " << tensor.toString()
              << " , device " << tensor.device() << " " << tensor.key_set();
  return impl;
}

void cuda_fallback(const c10::OperatorHandle &op, torch::jit::Stack *stack) {
  auto schema = op.schema();
  DLOG(INFO) << "redispatching " << schema << " to CUDA" << std::endl;
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
  auto handle = sta::Empty(size, dlpack_dtype, sta::MemType::kTrain);
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
  auto handle = sta::EmptyStrided(size.vec(), stride.vec(), dlpack_dtype, sta::MemType::kTrain);
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

const at::Tensor & as_strided_(
    const at::Tensor & self, at::IntArrayRef size, at::IntArrayRef stride, 
    c10::optional<int64_t> storage_offset) {
  auto impl = GetColTensorImpl(self);
  sta::AsStrided_(impl->Handle(), size, stride, storage_offset);
  const_cast<ColTensorImpl*>(impl)->UpdateAll();
  return self;
}
    

at::Tensor _reshape_alias(const at::Tensor& self, at::IntArrayRef size, at::IntArrayRef stride) {
  auto impl = GetColTensorImpl(self);
  return MakeColTensorAlias(sta::AsStrided(
      impl->Handle(), size, stride, impl->storage_offset()), self);
}

const at::Tensor& resize_(const at::Tensor& self, at::IntArrayRef size, c10::optional<at::MemoryFormat> memory_format) {
  auto impl = GetColTensorImpl(self);
  auto tensor = impl->Tensor();
  tensor.Resize(size, c10::nullopt);
  // LOG(INFO) << "resize_ " << "self " << impl  << " -> " << self.unsafeGetTensorImpl() << " "
  //           << size << " new ts " << self.sizes() << " " << self.numel() << " "
  //           << self.data_ptr();
  // impl->set_sizes_and_strides(tensor.Shape(), tensor.Stride());
  const_cast<ColTensorImpl*>(impl)->UpdateAll();
  return self;
}

at::Tensor view(const at::Tensor& self, at::IntArrayRef size) {
  // DLOG(INFO) << "view " << size << " " << self.sizes() << " " << self.numel();
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


at::Tensor nonzero(const at::Tensor & self) {
  c10::optional<at::Device> common_device = at::nullopt;
(void)common_device; // Suppress unused variable warning

  c10::impl::check_and_update_common_device(common_device, self, "nonzero", "self");

  const at::OptionalDeviceGuard device_guard(device_of(self));

  at::Tensor out = MakeColTensorEmpty({0}, self.options().dtype(at::kLong));
  // return at::native::nonzero_out_cuda(self, out);
  return nonzero_out_cuda(self, out);
}

at::Tensor & set_source_Storage_storage_offset(
  at::Tensor & self, at::Storage source, 
  int64_t storage_offset, at::IntArrayRef size, at::IntArrayRef stride) {
  LOG(FATAL) << "set tensor with a storage is unsupported op";
}

at::Tensor & set_source_Tensor(at::Tensor & self, const at::Tensor & source) {
  auto self_impl = GetColTensorImpl(self);
  auto source_impl = GetColTensorImpl(source);
  if (self_impl == source_impl) {
    return self;
  }
  auto self_tensor = self_impl->Tensor();
  auto source_tensor = source_impl->Tensor();
  sta::CheckMemoryBound(self_tensor.Shape(), self_tensor.Stride(), 
      self_tensor->dtype, source_tensor.StorageOffset(), source_tensor.MData());
  auto stride = source_tensor.Stride();
  at::OptionalIntArrayRef stride_opt = stride.data() != nullptr ?
                                          at::OptionalIntArrayRef(stride) : c10::nullopt;
  auto size = source_tensor.Shape();
  self_tensor.SetByteOffset(source_tensor->byte_offset);
  self_tensor.Resize(size, stride_opt);
  self_impl->UpdateAll();
  return self;
}

// cudnn
at::Tensor cudnn_convolution(
    const at::Tensor & self, const at::Tensor & weight, 
    at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, 
    int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
  c10::optional<at::Device> common_device = c10::nullopt;
(void)common_device; // Suppress unused variable warning

  c10::impl::check_and_update_common_device(common_device, self, "cudnn_convolution", "self");
  c10::impl::check_and_update_common_device(common_device, weight, "cudnn_convolution", "weight");

  const at::OptionalDeviceGuard device_guard(device_of(self));
  return cudnn::cudnn_convolution_custom(
      self, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
}

at::Tensor cudnn_convolution_transpose(
    const at::Tensor & self, const at::Tensor & weight, 
    at::IntArrayRef padding, at::IntArrayRef output_padding, 
    at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, 
    bool benchmark, bool deterministic, bool allow_tf32) {
  c10::optional<at::Device> common_device = c10::nullopt;
(void)common_device; // Suppress unused variable warning

  c10::impl::check_and_update_common_device(common_device, self, "cudnn_convolution_transpose", "self");
  c10::impl::check_and_update_common_device(common_device, weight, "cudnn_convolution_transpose", "weight");

  const at::OptionalDeviceGuard device_guard(device_of(self));
  return cudnn::cudnn_convolution_transpose_custom(
      self, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
}

at::Tensor cudnn_convolution_relu(
    const at::Tensor & self, const at::Tensor & weight, 
    const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, 
    at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
  c10::optional<at::Device> common_device = c10::nullopt;
(void)common_device; // Suppress unused variable warning

  c10::impl::check_and_update_common_device(common_device, self, "cudnn_convolution_relu", "self");
  c10::impl::check_and_update_common_device(common_device, weight, "cudnn_convolution_relu", "weight");
  c10::impl::check_and_update_common_device(common_device, bias, "cudnn_convolution_relu", "bias");

  const at::OptionalDeviceGuard device_guard(device_of(self));
  return cudnn::cudnn_convolution_relu_custom(
      self, weight, bias, stride, padding, dilation, groups);
}


at::Tensor cudnn_convolution_add_relu(
    const at::Tensor & self, const at::Tensor & weight, 
    const at::Tensor & z, const c10::optional<at::Scalar> & alpha, 
    const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, 
    at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
  c10::optional<at::Device> common_device = c10::nullopt;
(void)common_device; // Suppress unused variable warning

  c10::impl::check_and_update_common_device(common_device, self, "cudnn_convolution_add_relu", "self");
  c10::impl::check_and_update_common_device(common_device, weight, "cudnn_convolution_add_relu", "weight");
  c10::impl::check_and_update_common_device(common_device, z, "cudnn_convolution_add_relu", "z");
  c10::impl::check_and_update_common_device(common_device, bias, "cudnn_convolution_add_relu", "bias");

  const at::OptionalDeviceGuard device_guard(device_of(self));
  return cudnn::cudnn_convolution_add_relu_custom(
      self, weight, z, alpha, bias, stride, padding, dilation, groups);
}


TORCH_LIBRARY_IMPL(aten, CUDA, m) {
  m.impl("empty.memory_format", TORCH_FN(empty));
  m.impl("empty_strided", TORCH_FN(empty_strided));
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  // RegisterCUDA
  m.impl("as_strided", TORCH_FN(as_strided));
  m.impl("_reshape_alias", TORCH_FN(_reshape_alias));
  m.impl("resize_", TORCH_FN(resize_));
  m.impl("view", TORCH_FN(view));

  m.impl("set_.source_Storage_storage_offset", TORCH_FN(set_source_Storage_storage_offset));
  m.impl("set_.source_Tensor", TORCH_FN(set_source_Tensor));

  m.impl("nonzero", TORCH_FN(nonzero));

  // RegisterCompositeExplicitAutograd
  m.impl("_unsafe_view", TORCH_FN(view));
  m.impl("view.dtype", TORCH_FN(view_dtype));
  m.impl("alias", TORCH_FN(alias));
  m.impl("as_strided_", TORCH_FN(as_strided_));


  // convolution
  // m.impl("convolution_overrideable", convolution);
  m.impl("_convolution", TORCH_FN(_convolution));
  m.impl("convolution_backward", TORCH_FN(convolution_backward));

  // cudnn
  m.impl("cudnn_convolution", TORCH_FN(cudnn_convolution));
  m.impl("cudnn_convolution_transpose", TORCH_FN(cudnn_convolution_transpose));
  m.impl("cudnn_convolution_relu", TORCH_FN(cudnn_convolution_relu));
  m.impl("cudnn_convolution_add_relu", TORCH_FN(cudnn_convolution_add_relu));

  // avoid use default kernel (e.g., CompositeExplicitAutograd and CompositeImplicitAutograd)
  m.impl("addr", torch::CppFunction::makeFromBoxedFunction<
      &cuda_fallback>());
  m.impl("addr.out", torch::CppFunction::makeFromBoxedFunction<
      &cuda_fallback>());
  m.impl("native_group_norm", torch::CppFunction::makeFromBoxedFunction<
      &cuda_fallback>());
  m.impl("native_layer_norm", torch::CppFunction::makeFromBoxedFunction<
      &cuda_fallback>());
  m.impl("is_pinned", torch::CppFunction::makeFromBoxedFunction<
      &cuda_fallback>());
  m.impl("mish_backward", torch::CppFunction::makeFromBoxedFunction<
      &cuda_fallback>());  
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<
      &cuda_fallback>());
}

struct ColTensorInitializer {
  ColTensorInitializer() {
    LOG(INFO) << "ColTensor Initialized";
    auto has_server_env = std::getenv("SHARED_TENSOR_HAS_SERVER");
    auto pool_size_env = std::getenv("SHARED_TENSOR_POOL_GB");
    bool has_server = has_server_env && std::string(has_server_env) == "1";
    double pool_gb = 12;
    if (!has_server && !pool_size_env) {
      LOG(INFO) << "SHARED_TENSOR_POOL_GB not set, use default 12GB";
    } else if (pool_size_env) {
      pool_gb = std::stod(pool_size_env);
    }
    size_t pool_nbytes = static_cast<size_t>(pool_gb * 1024 * 1024 * 1024);
    colserve::sta::Init(pool_nbytes, !has_server);
  }
};

static ColTensorInitializer col_tensor_initializer __attribute__ ((init_priority (101)));

}
}