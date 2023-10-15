#ifndef COLSERVE_CUDNN_CUSTOM_H
#define COLSERVE_CUDNN_CUSTOM_H

#include <ATen/Tensor.h>

namespace torch_col { namespace cudnn {

at::Tensor cudnn_convolution_custom(
    const at::Tensor& input_t, const at::Tensor& weight_t,
    at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic, bool allow_tf32);

at::Tensor cudnn_convolution_transpose_custom(
    const at::Tensor& input, const at::Tensor& weight,
    at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic, bool allow_tf32);

at::Tensor cudnn_convolution_relu_custom(
    const at::Tensor& input_t,
    const at::Tensor& weight_t,
    const c10::optional<at::Tensor>& bias_t,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups);

at::Tensor cudnn_convolution_add_relu_custom(
    const at::Tensor& input_t,
    const at::Tensor& weight_t,
    const at::Tensor& z_t,
    const c10::optional<at::Scalar>& alpha,
    const c10::optional<at::Tensor>& bias_t,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups);

}}


#endif