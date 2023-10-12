#ifndef COLSERVE_CONVOLUTION_H
#define COLSERVE_CONVOLUTION_H

#include <ATen/native/ConvUtils.h>
#include <ATen/core/grad_mode.h>
#include <c10/util/Exception.h>

#include <iostream>

namespace torch_col {

std::ostream& operator<<(std::ostream& os, at::native::ConvBackend backend);

struct ConvParamsCustom {
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  bool transposed;
  std::vector<int64_t> output_padding;
  int groups;
  bool benchmark;
  bool deterministic;
  bool cudnn_enabled;
  bool allow_tf32;

  bool is_strided() const {
    bool is_strided = false;
    for (auto s : stride) {
      is_strided |= (s != 1);
    }
    return is_strided;
  }
  bool is_dilated() const {
    bool is_dilated = false;
    for (auto d : dilation) {
      is_dilated |= (d != 1);
    }
    return is_dilated;
  }
  bool is_padded() const {
    bool is_padded = false;
    for (auto p : padding) {
      is_padded |= (p != 0);
    }
    return is_padded;
  }
  bool is_output_padding_neg() const {
    bool is_non_neg = false;
    for (auto p : output_padding) {
      is_non_neg |= (p < 0);
    }
    return is_non_neg;
  }
  bool is_output_padding_big() const {
    bool is_big = false;
    for (auto i: c10::irange(output_padding.size())) {
      is_big |= (output_padding[i] >= stride[i]);
    }
    return is_big;
  }
  bool is_padding_neg() const {
    bool is_non_neg = false;
    for (auto p : padding) {
      is_non_neg |= (p < 0);
    }
    return is_non_neg;
  }
  bool is_stride_nonpos() const {
    bool is_nonpos = false;
    for (auto s : stride) {
      is_nonpos |= (s <= 0);
    }
    return is_nonpos;
  }
  void view1d_as_2d() {
    if (stride.size() == 1) {
      stride.insert(stride.begin(), 1);
      padding.insert(padding.begin(), 0);
      dilation.insert(dilation.begin(), 1);
      output_padding.insert(output_padding.begin(), 0);
    }
  }
  bool use_cpu_depthwise3x3_winograd(const at::Tensor& input, const at::Tensor& weight) const {
    return false;
  }
  bool needs_64bit_indexing_no_split(const at::Tensor& input, const at::Tensor& weight) const;
  bool use_cudnn(const at::Tensor& input, const at::Tensor& weight) const;
  bool use_cudnn_depthwise(const at::Tensor& input, const at::Tensor& weight) const;
  bool use_miopen(const at::Tensor& input, const at::Tensor& weight, bool bias_defined) const {
    return false;
  }
  bool use_mkldnn(const at::Tensor& input, const at::Tensor& weight) const {
    return false;
  }
  bool use_nnpack(const at::Tensor& input, const at::Tensor& weight) const {
    return false;
  }
  bool use_xnnpack(const at::Tensor& input, const at::Tensor& weight,
                   const at::OptionalIntArrayRef bias_sizes_opt) const {
    return false;
  }
  bool use_mps(const at::Tensor& input, const at::Tensor& weight) const {
    return false;
  }
  bool is_depthwise(const at::Tensor& input, const at::Tensor& weight) const;


};

at::Tensor _convolution(
    const at::Tensor& input_r, const at::Tensor& weight_r, const c10::optional<at::Tensor>& bias_r_opt,
    at::IntArrayRef stride_, at::IntArrayRef padding_, at::IntArrayRef dilation_,
    bool transposed_, at::IntArrayRef output_padding_, int64_t groups_,
    bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32);


}

#endif