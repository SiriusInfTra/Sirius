#include "ConvShared.h"
#include "../tensor_impl.h"
#include "cudnn_custom.h"

namespace torch_col { namespace cudnn {

// ---------------------------------------------------------------------
//
// Convolution forward / Transposed convolution backward
//
// ---------------------------------------------------------------------

at::Tensor cudnn_convolution_forward_custom(
    at::CheckedFrom c,
    const at::TensorArg& input, const at::TensorArg& weight,
    at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32)
{
  at::checkAllSameType(c, {input, weight});
  at::checkAllSameGPU(c, {input, weight});

  auto memory_format = at::native::cudnn_conv_suggest_memory_format(*input, *weight);
  // at::Tensor output_t = at::detail::empty_cuda(
  //     conv_output_size(input->sizes(), weight->sizes(),
  //                      padding, stride, dilation),
  //     input->options().memory_format(memory_format));
  at::Tensor output_t = MakeColTensorEmpty(
    at::native::conv_output_size(input->sizes(), weight->sizes(),
                       padding, stride, dilation),
    input->options().memory_format(memory_format)); 

  if (output_t.numel() == 0) {
    return output_t;
  }

  // Avoid ambiguity of "output" when this is being used as backwards
  at::TensorArg output{ output_t, "result", 0 };
  at::native::convolution_shape_check(c, input, weight, output, padding, stride, dilation, groups);

  at::Tensor weight_contig = weight->contiguous(memory_format);
  at::Tensor input_contig = input->contiguous(memory_format);

  at::native::raw_cudnn_convolution_forward_out(
      *output, input_contig, weight_contig,
      padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);

  return *output;
}

at::Tensor cudnn_convolution_custom(
    const at::Tensor& input_t, const at::Tensor& weight_t,
    at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
  at::TensorArg input  { input_t,  "input",  1 },
                weight { weight_t, "weight", 2 };
  at::CheckedFrom c = "cudnn_convolution_custom";
  auto output_t = cudnn_convolution_forward_custom(
    c, input, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
  return output_t;
}

at::Tensor cudnn_convolution_transpose_backward_input_custom(
    const at::Tensor& grad_output_t, const at::Tensor& weight_t,
    at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic, bool allow_tf32)
{
  at::TensorArg grad_output { grad_output_t,  "grad_output", 1 },
            weight      { weight_t, "weight", 2 };
  return cudnn_convolution_forward_custom(
    "cudnn_convolution_transpose_backward_input",
    grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
}


// ---------------------------------------------------------------------
//
// Convolution backward / Transposed convolution forward
//
// ---------------------------------------------------------------------

at::Tensor cudnn_convolution_backward_input_custom(
    at::CheckedFrom c,
    at::IntArrayRef input_size, const at::TensorArg& grad_output, const at::TensorArg& weight,
    at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32)
{
  at::checkAllSameType(c, {grad_output, weight});
  at::checkAllSameGPU(c, {grad_output, weight});

  auto memory_format = at::native::cudnn_conv_suggest_memory_format(*grad_output, *weight);
  // Tensor grad_input_t = at::detail::empty_cuda(
  //     input_size, grad_output->options().memory_format(memory_format));
  at::Tensor grad_input_t = MakeColTensorEmpty(
    input_size, grad_output->options().memory_format(memory_format));

  // Avoid "grad_input" when this is being used as transposed convolution
  at::TensorArg grad_input{ grad_input_t, "result", 0 };
  at::native::convolution_shape_check(c, grad_input, weight, grad_output, padding, stride, dilation, groups);

  at::Tensor weight_contig = weight->contiguous(memory_format);
  at::Tensor grad_output_contig = grad_output->contiguous(memory_format);

  at::native::raw_cudnn_convolution_backward_input_out(
      *grad_input, grad_output_contig, weight_contig,
      padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);

  return *grad_input;
}

at::Tensor cudnn_convolution_transpose_forward_custom(
    at::CheckedFrom c,
    const at::TensorArg& grad_output, const at::TensorArg& weight,
    at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32)
{
  auto input_size = at::native::conv_input_size(grad_output->sizes(), weight->sizes(),
                                    padding, output_padding, stride, dilation, groups);
  return cudnn_convolution_backward_input_custom(c, input_size, grad_output, weight,
                                    padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
}

at::Tensor cudnn_convolution_backward_input_custom(
    at::IntArrayRef input_size, const at::Tensor& grad_output_t, const at::Tensor& weight_t,
    at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32)
{
  at::TensorArg grad_output{ grad_output_t, "grad_output", 1 },
                weight{ weight_t, "weight", 2 };
  return cudnn_convolution_backward_input_custom(
      "cudnn_convolution_backward_input_custom",
      input_size, grad_output, weight,
      padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
}

at::Tensor cudnn_convolution_transpose_custom(
    const at::Tensor& input_t, const at::Tensor& weight_t,
    at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic, bool allow_tf32)
{
  at::TensorArg input  { input_t,  "input",  1 },
                weight { weight_t, "weight", 2 };
  at::CheckedFrom c = "cudnn_convolution_transpose_custom";
  auto output_t = cudnn_convolution_transpose_forward_custom(
    c, input, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
  return output_t;
}

// ---------------------------------------------------------------------
//
// Convolution backward (weight)
//
// ---------------------------------------------------------------------

at::Tensor cudnn_convolution_backward_weight(
    at::CheckedFrom c,
    at::IntArrayRef weight_size, const at::Tensor& grad_output_t, const at::Tensor& input_t,
    at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32)
{
  auto layout = at::native::cudnn_conv_suggest_memory_format(input_t, grad_output_t);

  at::Tensor grad_output_contig_t = grad_output_t.contiguous(layout);
  at::TensorArg grad_output_contig{ grad_output_contig_t, "grad_output", 1 };

  at::Tensor input_contig_t = input_t.contiguous(layout);
  at::TensorArg input{ input_contig_t, "input", 2};

  at::checkAllSameType(c, {grad_output_contig, input});
  at::checkAllSameGPU(c, {grad_output_contig, input});

  auto grad_weight_t = at::empty(weight_size, grad_output_contig->options(), layout);

  // For uniformity with everything else, although it seems grad_weight
  // would be unambiguous too.
  at::TensorArg grad_weight{ grad_weight_t, "result", 0 };
  at::native::convolution_shape_check(c, input, grad_weight, grad_output_contig, padding, stride, dilation, groups);

  at::native::raw_cudnn_convolution_backward_weight_out(
      *grad_weight, *grad_output_contig, *input,
      padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);

  return grad_weight_t;
}

at::Tensor cudnn_convolution_backward_weight(
    at::IntArrayRef weight_size,
    const at::Tensor& grad_output_t,
    const at::Tensor& input_t,
    at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32)
{
  return cudnn_convolution_backward_weight(
      "cudnn_convolution_backward_weight",
      weight_size, grad_output_t, input_t,
      padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
}

std::tuple<at::Tensor,at::Tensor> cudnn_convolution_backward_custom(
    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& weight,
    at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32, std::array<bool,2> output_mask) {

  at::Tensor grad_output = grad_output_t.contiguous(input.suggest_memory_format());

  at::Tensor grad_input, grad_weight;
  if (input.numel() == 0) {
    if (output_mask[0]) {
      grad_input = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    }
    if (output_mask[1]) {
      grad_weight = at::zeros_like(weight, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    }
  } else {
    if (output_mask[0]) {
      grad_input = cudnn_convolution_backward_input_custom(input.sizes(), grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
    }
    if (output_mask[1]) {
      grad_weight = cudnn_convolution_backward_weight(weight.sizes(), grad_output, input, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
    }
  }

  return std::tuple<at::Tensor,at::Tensor>{grad_input, grad_weight};
}

at::Tensor cudnn_convolution_transpose_backward_weight(
    at::IntArrayRef weight_size,
    const at::Tensor& grad_output_t,
    const at::Tensor& input_t,
    at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32)
{
  return cudnn_convolution_backward_weight(
      "cudnn_convolution_backward_weight",
      weight_size, input_t, grad_output_t,
      padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
}

std::tuple<at::Tensor,at::Tensor> cudnn_convolution_transpose_backward_custom(
    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& weight,
    at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32, std::array<bool,2> output_mask) {

  at::Tensor grad_output = grad_output_t.contiguous(input.suggest_memory_format());

  at::Tensor grad_input, grad_weight;
  if (output_mask[0]) {
    grad_input = cudnn_convolution_transpose_backward_input_custom(grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
  }
  if (output_mask[1]) {
    grad_weight = cudnn_convolution_transpose_backward_weight(weight.sizes(), grad_output, input, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
  }

  return std::tuple<at::Tensor,at::Tensor>{grad_input, grad_weight};
}

at::Tensor cudnn_convolution_relu_custom(
    const at::Tensor& input_t,
    const at::Tensor& weight_t,
    const c10::optional<at::Tensor>& bias_t,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
  auto memory_format = at::native::cudnn_conv_suggest_memory_format(input_t, weight_t);
  const at::Tensor input = input_t.contiguous(memory_format);
  const at::Tensor weight = weight_t.contiguous(memory_format);

  // FuseFrozenConvAddRelu performs some tensor shape checking
  // Tensor output_t = at::detail::empty_cuda(
  //     conv_output_size(
  //         input.sizes(), weight.sizes(), padding, stride, dilation),
  //     input.options().memory_format(memory_format));
  at::Tensor output_t = MakeColTensorEmpty(
      at::native::conv_output_size(
          input.sizes(), weight.sizes(), padding, stride, dilation),
      input.options().memory_format(memory_format));
  if (output_t.numel() == 0) {
    return output_t;
  }

  auto& ctx = at::globalContext();
  bool benchmark = ctx.benchmarkCuDNN();
  bool allow_tf32 = ctx.allowTF32CuDNN();
  auto _bias = bias_t.has_value()
          ? bias_t.value()
          : at::zeros(
                {output_t.size(1)},
                optTypeMetaToScalarType(output_t.options().dtype_opt()),
                output_t.options().layout_opt(),
                output_t.options().device_opt(),
                output_t.options().pinned_memory_opt());

#ifdef AT_CUDNN_CONV_BIAS_RELU_FALLBACK
  raw_cudnn_convolution_add_relu_fallback_out(
      output_t,
      input,
      weight,
      output_t, // use output_t as z to satisfy CUDNN API
      0, // alpha
      _bias,
      stride,
      padding,
      dilation,
      groups,
      benchmark, // benchmark
      false, // deterministic
      allow_tf32  // allow_tf32
  );
#else  // AT_CUDNN_CONV_BIAS_RELU_FALLBACK
  at::native::raw_cudnn_convolution_add_relu_out(
      output_t,
      input,
      weight,
      output_t, // use output_t as z to satisfy CUDNN API
      0, // alpha
      _bias,
      stride,
      padding,
      dilation,
      groups,
      benchmark, // benchmark
      false, // deterministic
      allow_tf32  // allow_tf32
  );
#endif

  return output_t;
}

at::Tensor cudnn_convolution_add_relu_custom(
    const at::Tensor& input_t,
    const at::Tensor& weight_t,
    const at::Tensor& z_t,
    const c10::optional<at::Scalar>& alpha,
    const c10::optional<at::Tensor>& bias_t,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
  auto memory_format = at::native::cudnn_conv_suggest_memory_format(input_t, weight_t);
  const at::Tensor input = input_t.contiguous(memory_format);
  const at::Tensor weight = weight_t.contiguous(memory_format);
  at::Tensor z = z_t;
  if (z.suggest_memory_format() != memory_format) {
    z = z.to(memory_format);
  }
  z = z.contiguous(memory_format);

  // FuseFrozenConvAddRelu performs some tensor shape checking
  // Tensor output_t = at::detail::empty_cuda(
  //     conv_output_size(
  //         input.sizes(), weight.sizes(), padding, stride, dilation),
  //     input.options().memory_format(memory_format));
  at::Tensor output_t = MakeColTensorEmpty(
      at::native::conv_output_size(
          input.sizes(), weight.sizes(), padding, stride, dilation),
      input.options().memory_format(memory_format));
  if (output_t.numel() == 0) {
    return output_t;
  }

  auto& ctx = at::globalContext();
  bool allow_tf32 = ctx.allowTF32CuDNN();
  bool benchmark = ctx.benchmarkCuDNN();
  auto _alpha = alpha.has_value() ? alpha.value().to<float>() : 1.0;
  auto _bias = bias_t.has_value()
          ? bias_t.value()
          : at::zeros(
                {output_t.size(1)},
                optTypeMetaToScalarType(output_t.options().dtype_opt()),
                output_t.options().layout_opt(),
                output_t.options().device_opt(),
                output_t.options().pinned_memory_opt());

#ifdef AT_CUDNN_CONV_BIAS_RELU_FALLBACK
  raw_cudnn_convolution_add_relu_fallback_out(
      output_t,
      input,
      weight,
      z,
      _alpha,
      _bias,
      stride,
      padding,
      dilation,
      groups,
      benchmark,
      false, // deterministic
      allow_tf32  // allow_tf32
  );
#else  // AT_CUDNN_CONV_BIAS_RELU_FALLBACK
  at::native::raw_cudnn_convolution_add_relu_out(
      output_t,
      input,
      weight,
      z,
      _alpha,
      _bias,
      stride,
      padding,
      dilation,
      groups,
      benchmark,
      false, // deterministic
      allow_tf32  // allow_tf32
  );
#endif  // AT_CUDNN_CONV_BIAS_RELU_FALLBACK

  return output_t;
}


using namespace at::native;
REGISTER_CUDA_DISPATCH(cudnn_convolution_backward_stub, &cudnn_convolution_backward_custom);
REGISTER_CUDA_DISPATCH(cudnn_convolution_transpose_backward_stub, &cudnn_convolution_transpose_backward_custom);


}}