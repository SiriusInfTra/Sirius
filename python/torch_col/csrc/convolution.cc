#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <c10/util/Exception.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/ConvolutionMM3d.h>

#include "convolution.h"
#include "utils.h"

#include <undef_log.h>
#include <glog/logging.h>

namespace torch_col {

std::ostream& operator<<(std::ostream& os, at::native::ConvBackend backend) {
  switch (backend)
  {
  case at::native::ConvBackend::CudaDepthwise2d:
    os << "CudaDepthwise2d"; return os;
  case at::native::ConvBackend::CudaDepthwise3d:
    os << "CudaDepthwise3d"; return os;
  case at::native::ConvBackend::Cudnn:
    os << "Cudnn"; return os;
  case at::native::ConvBackend::CudnnTranspose:
    os << "CudnnTranspose"; return os;
  case at::native::ConvBackend::Slow2d:
    os << "Slow2d"; return os;
  case at::native::ConvBackend::Slow3d:
    os << "Slow3d"; return os;
  case at::native::ConvBackend::SlowDilated2d:
    os << "SlowDilated2d"; return os;
  case at::native::ConvBackend::SlowDilated3d:
    os << "SlowDilated3d"; return os;
  case at::native::ConvBackend::SlowTranspose2d:
    os << "SlowTranspose2d"; return os;
  case at::native::ConvBackend::SlowTranspose3d:
    os << "SlowTranspose3d"; return os;
  case at::native::ConvBackend::Empty:
    os << "Empty"; return os;
  default:
    TORCH_CHECK(false, "torch_col unsupport backend: ", static_cast<int>(backend));
  }
}

bool ConvParamsCustom::needs_64bit_indexing_no_split(const at::Tensor& input, const at::Tensor& weight) const {
  constexpr int64_t int_max = std::numeric_limits<int>::max();
  int64_t numel_input = input.numel();
  // empty input
  if (numel_input == 0) {
    return false;
  }
  // input size can not be reduced to the range of int by splitting the batch dim
  int64_t n = input.size(0);
  if (numel_input / n > int_max) {
    return true;
  }
  // output size can not be reduced to the range of int by splitting the batch dim
  int64_t outsize = 1;
  if (transposed) {
    std::vector<int64_t> o = at::native::conv_input_size(input.sizes(), weight.sizes(), padding, output_padding, stride, dilation, groups);
    outsize = c10::multiply_integers(o.begin() + 1, o.end());
  } else {
    std::vector<int64_t> o = at::native::conv_output_size(input.sizes(), weight.sizes(), padding, stride, dilation);
    outsize = c10::multiply_integers(o.begin() + 1, o.end());
  }
  return outsize > int_max;
}

bool ConvParamsCustom::use_cudnn(const at::Tensor& input, const at::Tensor& weight) const {
  if (needs_64bit_indexing_no_split(input, weight)) {
    return false;
  }
  if (!at::detail::getCUDAHooks().compiledWithCuDNN()) {
    return false;
  }
  if (!input.device().is_cuda() || !cudnn_enabled) {
    return false;
  }
  if (input.scalar_type() == at::kBFloat16 || weight.scalar_type() == at::kBFloat16)  {
    return at::native::cudnnv8_enabled_check_debug();
  }
  if (at::native::cudnn_conv_suggest_memory_format(input, weight) == at::MemoryFormat::Contiguous) {
    // bypass dilation checks for channels_last convolution
    if (deterministic && is_dilated()) {
      // cudnn doesn't support deterministic dilated convolution fully yet
      return false;
    }
    if (is_dilated()) {
      return at::detail::getCUDAHooks().supportsDilatedConvolutionWithCuDNN() && !is_output_padding_big();
    }
  }
  return !is_output_padding_big();

}

// Check workload to activate fast depthwise FP16 cudnn conv kernels
static bool check_cudnn_depthwise_workload(const at::Tensor& input, int stride) {
  int w = input.size(3);  // same as h
  int ch = input.size(1);
  int bs = input.size(0);
  if (stride==1) {
    if (w >= 7) {
      // All batch sizes and nb_channels
      if (w >= 112) {
        return true;
      }

      // large nb_channels
      if (ch >= 1024) {
        // NOLINTNEXTLINE(bugprone-branch-clone,cppcoreguidelines-avoid-magic-numbers)
        if (w >= 56) {
          return true;
        } else if (bs >= 32) {
          return true;
        }
      }

      // batch_size specific
      if (bs >= 128) {
        // NOLINTNEXTLINE(bugprone-branch-clone,cppcoreguidelines-avoid-magic-numbers)
        if (ch >= 512) {
          return true;
        } else if (ch >= 64) {
          if (w >= 14) {
            return true;
          }
        } else if ((ch >= 32) && (w >=28)) {
          return true;
        }
      } else if (bs >= 64) {
        // NOLINTNEXTLINE(bugprone-branch-clone,cppcoreguidelines-avoid-magic-numbers)
        if ((ch >= 256) && (w >= 14)) {
          return true;
        } else if ((ch >= 32) && (w >= 28)) {
          return true;
        }
      } else if (bs >= 32) {
        // NOLINTNEXTLINE(bugprone-branch-clone,cppcoreguidelines-avoid-magic-numbers)
        if ((ch >= 256) && (w >= 14)) {
          return true;
        } else if ((ch >= 128) && (w >= 28)) {
          return true;
        } else if ((ch >= 32) && (w >= 56)) {
          return true;
        }
      } else if (bs >= 16) {
        if ((ch >= 1024) && (w >= 14)) {
          return true;
        }
        // NOLINTNEXTLINE(bugprone-branch-clone,cppcoreguidelines-avoid-magic-numbers)
        if ((ch >= 256) && (w >= 28)) {
          return true;
        } else if ((ch >= 32) && (w >= 56)) {
          return true;
        }
      } else if (bs >= 8) {
        // NOLINTNEXTLINE(bugprone-branch-clone,cppcoreguidelines-avoid-magic-numbers)
        if ((ch >= 512) && (w >= 28)) {
          return true;
        } else if ((ch >= 64) && (w >= 56)) {
          return true;
        }
      }
    }
  } else if (stride==2) {
    if (ch < 256) {
      return false;
    }

    if (w >= 7) {
      if (bs >= 128) {
        // NOLINTNEXTLINE(bugprone-branch-clone,cppcoreguidelines-avoid-magic-numbers)
        if (ch >= 1024) {
          return true;
        } else if ((ch >= 512) && (w >= 14)) {
          return true;
        } else if (w >= 28) {
          return true;
        }
      } else if (bs >= 64) {
        // NOLINTNEXTLINE(bugprone-branch-clone,cppcoreguidelines-avoid-magic-numbers)
        if ((ch >= 512) && (w >= 14)) {
          return true;
        } else if (w >= 28) {
          return true;
        }
      } else if (bs >= 32) {
        // NOLINTNEXTLINE(bugprone-branch-clone,cppcoreguidelines-avoid-magic-numbers)
        if ((ch >= 1024) && (w >= 14)) {
          return true;
        } else if (w >= 28) {
          return true;
        }
      } else if (bs >= 16) {
        // NOLINTNEXTLINE(bugprone-branch-clone,cppcoreguidelines-avoid-magic-numbers)
        if ((ch >= 512) && (w >= 28)) {
          return true;
        } else if (w >= 56) {
          return true;
        }
      } else if (bs >= 8) {
        // NOLINTNEXTLINE(bugprone-branch-clone,cppcoreguidelines-avoid-magic-numbers)
        if ((ch >= 1024) && (w >= 28)) {
          return true;
        } else if (w >= 56) {
          return true;
        }
      } else if (bs >= 1) {
        if ((ch >= 512) && (w >=112)) {
          return true;
        }
      }
    }
  }
  return false;
}


// simplified version for cudnn 8.2 and above
static bool check_cudnn_depthwise_workload_with_filter(const at::Tensor& input, int stride, const at::Tensor& weight) {
  // 1D conv
  if(input.size(2) == 1 && stride == 1){
    return true;
  }

  // 2d conv
  // only square filters
  if (weight.size(2) != weight.size(3)) return false;
  int filter = weight.size(3);
  // only 1/3/5 filter
  if (filter != 1 && filter != 3 && filter != 5) return false;
  // we don't enforce square input but only check width to reduce heuristic space
  if (input.size(3) < 7) return false; // min width 7
  int w = input.size(3);
  // only 1/2 stride, use cudnn for all stride 1
  if (stride == 1) return true;
  if (stride != 2) return false;

  int ch = input.size(1);
  int bs = input.size(0);
  // special case since bs1 show good perf in lots of cases
  if (bs == 1) {
    if (filter == 1 && w <= 28) return true;
    if (filter == 3 || filter == 5) return true;
  } else {
    if (filter == 1 && bs <= 16 && ch >= 128 && w <= 7) return true;
    if (filter == 3 || filter == 5) {
      if ((ch >= 512) || (ch >= 256 && w >= 28)) return true;
    }
  }
  return false;
}

bool ConvParamsCustom::use_cudnn_depthwise(const at::Tensor& input, const at::Tensor& weight) const {
  if (at::native::cudnn_conv_suggest_memory_format(input, weight) 
      != at::MemoryFormat::Contiguous && use_cudnn(input, weight)) {
    // always use cudnn_depthwise for channels_last format
    return true;
  }
  if (at::detail::getCUDAHooks().supportsDepthwiseConvolutionWithCuDNN()) {
    long cudnn_version = at::detail::getCUDAHooks().versionCuDNN();
    if (cudnn_version >= 8200) {
      bool kernel_cond =  (use_cudnn(input, weight) &&
                           input.scalar_type() == at::kHalf && // only for FP16
                           weight.scalar_type() == at::kHalf &&
                           is_depthwise(input, weight) &&
                           input.ndimension() == 4 &&   // TODO: 5-D contiguous depthwise is not supported yet, need benchmarks
                           !is_dilated() && // no dilation supported
                           (stride[0] == stride[1] || input.size(2) == 1) && // square or 1d
                           input.size(1) >= 32); // min 32 channels supported)
      if (kernel_cond) {
        return check_cudnn_depthwise_workload_with_filter(input, stride[1], weight);
      }
    }
    // keep (7600 <= cudnn < 8200) code unchanged
    bool kernel_cond =  (cudnn_version >= 7600 &&
                         use_cudnn(input, weight) &&
                         input.scalar_type() == at::kHalf && // only for FP16
                         weight.scalar_type() == at::kHalf &&
                         is_depthwise(input, weight) &&
                         input.ndimension() == 4 &&   // TODO: 5-D contiguous depthwise is not supported yet, need benchmarks
                         weight.size(2) == weight.size(3) && // only square kernels
                         input.size(2) >= 7 && // min width/height 7
                         !is_dilated() && // no dilation supported
                         stride[0] == stride[1] && // equal strides
                         ((weight.size(3) == 3) || (weight.size(3) == 1)) &&
                         input.size(1) >= 32); // min 32 channels supported)
    if (kernel_cond) {
      return check_cudnn_depthwise_workload(input, stride[0]);
    } else {
      return false;
    }
  } else {
    return false;
  }
}


bool ConvParamsCustom::is_depthwise(const at::Tensor& input, const at::Tensor& weight) const {
  return input.device().is_cuda() &&
         !transposed &&
         (input.ndimension() == 4 || input.ndimension() == 5) &&
         input.size(1) == groups &&
         groups > 1 && // no point if there is only a single group
         weight.size(0) % input.size(1) == 0; // output channels must be a multiple of input channels

}

at::native::ConvBackend select_conv_backend_custom(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::OptionalIntArrayRef bias_sizes_opt,
    const bool need_backward,
    const ConvParamsCustom& params) {

  // don't send empty inputs through backends
  if (input.size(0) == 0 || input.size(1) == 0) {
    return at::native::ConvBackend::Empty;
  } else if (input.numel() == 0) {
    TORCH_CHECK(false, "Only zero batch or zero channel inputs are supported, but got input shape: ", input.sizes());
  }

  if (params.is_depthwise(input, weight)) {
    if (params.use_cudnn_depthwise(input, weight)) {
      return at::native::ConvBackend::Cudnn;
    } else {
      if (input.ndimension() == 4) {
        return at::native::ConvBackend::CudaDepthwise2d;
      } else if (input.ndimension() == 5) {
        return at::native::ConvBackend::CudaDepthwise3d;
      } else {
        // unsupported
      }
    }
  } else if (params.use_cudnn(input, weight)) {
    if (params.transposed) {
      return at::native::ConvBackend::CudnnTranspose;
    } else {
      return at::native::ConvBackend::Cudnn;
    }
  } else if (input.device().is_cuda()) {
    // backends without support for groups
    if (params.transposed) {
      if (input.ndimension() == 4) {
        return at::native::ConvBackend::SlowTranspose2d;
      } else if (input.ndimension() == 5) {
        return at::native::ConvBackend::SlowTranspose3d;
      } else {
        // unsupported
      }
    } else {  /* Not transposed */
      if (input.ndimension() == 4) {
        if (params.is_dilated()) {
          return at::native::ConvBackend::SlowDilated2d;
        } else {  /* dim == 4, non-dilated */
          return at::native::ConvBackend::Slow2d;
        }
      } else if (input.ndimension() == 5 && (input.device().is_cuda() || params.is_dilated())) {
        return at::native::ConvBackend::SlowDilated3d;
      } else if (input.ndimension() == 5) { /* dim == 5, CPU, non-dilated */
        /* CPU implementation has specialized MM kernels
           for non-dilated case here */
        return at::native::ConvBackend::Slow3d;
      } else {
        // unsupported
      }
    }
  } else {
    // Only reach here when input is backend with out-of-source implementation.
    return at::native::ConvBackend::Overrideable;
  }

  // Error out if no suitable backend was found.
  AT_ERROR("unsupported ConvNd parameters");
}

at::Tensor _convolution_nogroup_backend(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::native::ConvBackend backend,
    const ConvParamsCustom& params) {
  auto kernel_size = weight.sizes().slice(2);
  switch(backend) {
    case at::native::ConvBackend::NnpackSpatial:
      TORCH_INTERNAL_ASSERT(false, "NnpackSpatial backend was selected in PyTorch compiled without nnpack support");
    case at::native::ConvBackend::Slow2d:
      return at::thnn_conv2d(input, weight, kernel_size, bias, params.stride, params.padding);
    case at::native::ConvBackend::SlowDilated2d:
      return at::slow_conv_dilated2d(
          input, weight, kernel_size, bias, params.stride, params.padding, params.dilation);
    case at::native::ConvBackend::SlowDilated3d:
      return at::slow_conv_dilated3d(
          input, weight, kernel_size, bias, params.stride, params.padding, params.dilation);
    case at::native::ConvBackend::SlowTranspose2d:
      return at::slow_conv_transpose2d(
          input, weight, kernel_size, bias, params.stride, params.padding, params.output_padding, params.dilation);
    case at::native::ConvBackend::SlowTranspose3d:
      return at::slow_conv_transpose3d(
          input, weight, kernel_size, bias, params.stride, params.padding, params.output_padding, params.dilation);
    default:
      TORCH_CHECK(false, "Unsupported conv nogroup backend encountered");
  }
}


static void check_shape_forward(const at::Tensor& input,
                                const c10::IntArrayRef& weight_sizes, const at::Tensor& bias,
                                const ConvParamsCustom& params) {
  int64_t k = input.ndimension();
  int64_t weight_dim = weight_sizes.size();
  int64_t groups = params.groups;
  const auto& padding = params.padding;
  const auto& dilation = params.dilation;
  bool transposed = params.transposed;

  TORCH_CHECK(!params.is_padding_neg(), "negative padding is not supported");
  TORCH_CHECK(!params.is_output_padding_neg(), "negative output_padding is not supported");
  TORCH_CHECK(!params.is_stride_nonpos(), "non-positive stride is not supported");

  TORCH_CHECK(weight_dim == k,
           "Expected ", weight_dim, "-dimensional input for ", weight_dim,
           "-dimensional weight ", weight_sizes, ", but got ", k, "-dimensional input of size ",
           input.sizes(), " instead");
  TORCH_CHECK(weight_sizes[0] >= groups,
           "Given groups=", groups, ", expected weight to be at least ", groups,
           " at dimension 0, but got weight of size ", weight_sizes, " instead");
  TORCH_CHECK(weight_sizes[0] % groups == 0,
           "Given groups=", groups, ", expected weight to be divisible by ",
           groups, " at dimension 0, but got weight of size [", weight_sizes,
           "] instead");

  if (!transposed) {
    std::vector<int64_t> input_shape;
    std::vector<int64_t> kernel_shape;
    bool kernel_size_correct = true;

    TORCH_CHECK(input.size(1) == (weight_sizes[1] * groups),
                "Given groups=", groups, ", weight of size ", weight_sizes,
                ", expected input", input.sizes(), " to have ",
                (weight_sizes[1] * groups), " channels, but got ", input.size(1),
                " channels instead");

    TORCH_CHECK(!bias.defined() || (bias.ndimension() == 1 && bias.size(0) == weight_sizes[0]),
             "Given weight of size ", weight_sizes,
             ", expected bias to be 1-dimensional with ", weight_sizes[0], " elements",
             ", but got bias of size ", bias.sizes(), " instead");

    for (const auto i : c10::irange(2, k)) {
      input_shape.push_back(input.size(i) + 2 * padding[i-2]);
      // log new kernel size considering dilation
      kernel_shape.push_back(dilation[i-2] * (weight_sizes[i]-1) + 1);
      if (input_shape.back() < kernel_shape.back()) {
        kernel_size_correct = false;
      }
    }

    TORCH_CHECK(input_shape.size() == kernel_shape.size(), "Inconsistent shape between Input and Kernel");

    if (!kernel_size_correct) {
      // If kernel size is incorrect
      std::ostringstream input_ss;
      std::ostringstream kernel_ss;
      std::string separator = "";

      for (int i = 0, len = input_shape.size(); i < len; ++i) {
        input_ss << separator << input_shape[i];
        kernel_ss << separator << kernel_shape[i];
        separator = " x ";
      }

      AT_ERROR("Calculated padded input size per channel: (", input_ss.str(), "). "
               "Kernel size: (", kernel_ss.str(), "). Kernel size can't be greater than actual input size");
    }
  } else { // transposed
    TORCH_CHECK(input.size(1) == weight_sizes[0],
             "Given transposed=", transposed, ", weight of size ", weight_sizes,
             ", expected input", input.sizes(), " to have ", weight_sizes[0],
             " channels, but got ", input.size(1), " channels instead");
    TORCH_CHECK(!bias.defined() || (bias.ndimension() == 1 && bias.size(0) == weight_sizes[1] * groups),
             "Given transposed=", transposed, ", weight of size ", weight_sizes,
             ", expected bias to be 1-dimensional with ", weight_sizes[1] * groups, " elements",
             ", but got bias of size ", bias.sizes(), " instead");
  }
}

static void check_shape_backward(
    const at::Tensor& input,
    const c10::IntArrayRef& weight_sizes,
    const ConvParamsCustom& params) {
  check_shape_forward(input, weight_sizes, /*bias=*/ at::Tensor(), params);
}

static void check_input_same_type_as_parameters(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias) {
  TORCH_CHECK(input.options().type_equal(weight.options()),
      "Input type (", input.toString(), ") and weight type (", weight.toString(),
      ") should be the same");
  TORCH_CHECK(!bias.defined() || (input.options().type_equal(bias.options())),
      "Input type (", input.toString(), ") and bias type (", bias.toString(),
      ") should be the same");
}

static void check_input_same_type_as_parameters(
    const at::Tensor& input,
    const at::Tensor& weight) {
  check_input_same_type_as_parameters(input, weight, /*bias=*/ at::Tensor());
}

static void check_input_same_type_as_parameters(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::native::ConvBackend backend) {
  if (backend == at::native::ConvBackend::Mkldnn) {
    TORCH_CHECK(input.options().type_equal(weight.options())
        || (input.is_mkldnn() && weight.device().is_cpu() && weight.scalar_type() == at::kFloat),
        "Input type (", input.toString(), ") and weight type (", weight.toString(),
        ") should be the same or input should be a MKLDNN tensor and weight is a dense tensor");
    TORCH_CHECK(!bias.defined() || (input.options().type_equal(bias.options()))
        || (input.is_mkldnn() && bias.device().is_cpu() && bias.scalar_type() == at::kFloat),
        "Input type (", input.toString(), ") and bias type (", bias.toString(),
        ") should be the same or input should be a MKLDNN tensor and bias is a dense tensor");
  } else {
    check_input_same_type_as_parameters(input, weight, bias);
  }
}


static auto view4d(const at::Tensor& tensor) -> at::Tensor {
  TORCH_CHECK(tensor.ndimension() == 3,
           "expected 3D tensor, got tensor with ", tensor.ndimension(),
           " dimensions instead");
  return tensor.unsqueeze(2);
}

static auto view3d(const at::Tensor& tensor) -> at::Tensor {
  TORCH_CHECK(tensor.ndimension() == 4,
           "expected 4D tensor, got tensor with ", tensor.ndimension(),
           " dimensions instead");
  return tensor.squeeze(2);
}

static at::Tensor subtensor(at::Tensor& tensor, int dim, int groups, int g) {
  if (!tensor.defined()) {
    return at::Tensor();
  }
  const auto memory_format = tensor.suggest_memory_format();
  int64_t n = tensor.sizes()[dim] / groups;
  return tensor.narrow(dim, n * g, n).contiguous(memory_format);
}

static inline std::vector<int64_t> calc_output_size(
    const at::Tensor& input,
    const at::Tensor& weight,
    const ConvParamsCustom& params) {
  std::vector<int64_t> output_size = params.transposed ?
    at::native::conv_input_size(input.sizes(), weight.sizes(), params.padding, params.output_padding,
        params.stride, params.dilation, params.groups) :
    at::native::conv_output_size(input.sizes(), weight.sizes(), params.padding, params.stride, params.dilation);

  // Handle empty # of channels.
  if (input.size(1) == 0) {
    output_size[at::native::input_channels_dim] = 0;
  }
  return output_size;
}

static inline at::MemoryFormat determine_backend_memory_format(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::native::ConvBackend backend) {
  at::MemoryFormat backend_memory_format = at::MemoryFormat::Contiguous;
  auto k = weight.ndimension();
  switch(backend) {
    case at::native::ConvBackend::Cudnn:
    case at::native::ConvBackend::CudnnTranspose:
      if (at::detail::getCUDAHooks().compiledWithCuDNN()) {
        backend_memory_format = at::native::cudnn_conv_suggest_memory_format(input, weight);
      }
      break;
    case at::native::ConvBackend::Slow2d:
    case at::native::ConvBackend::SlowDilated2d:
      if (at::native::thnn_conv_use_channels_last(input, weight)) {
        backend_memory_format = at::MemoryFormat::ChannelsLast;
      }
      break;
    default:
      backend_memory_format = at::MemoryFormat::Contiguous;
  }
  return backend_memory_format;
}

at::Tensor _convolution(
    const at::Tensor& input_r, const at::Tensor& weight_r, const c10::optional<at::Tensor>& bias_r_opt,
    at::IntArrayRef stride_, at::IntArrayRef padding_, at::IntArrayRef dilation_,
    bool transposed_, at::IntArrayRef output_padding_, int64_t groups_,
    bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) {
  c10::MaybeOwned<at::Tensor> bias_r_maybe_owned = at::borrow_from_optional_tensor(bias_r_opt);
  const at::Tensor& bias_r = *bias_r_maybe_owned;

  auto input = input_r;
  auto weight = weight_r;
  auto bias = bias_r;
  auto k = weight.ndimension();
  c10::IntArrayRef weight_sizes = weight.sizes();
  int64_t dim = k - 2;

  TORCH_CHECK(dim > 0, "weight should have at least three dimensions");

  ConvParamsCustom params;
  params.stride = expand_param_if_needed(stride_, "stride", dim);
  params.padding = expand_param_if_needed(padding_, "padding", dim);
  params.dilation = expand_param_if_needed(dilation_, "dilation", dim);
  params.transposed = transposed_;
  params.output_padding = expand_param_if_needed(output_padding_, "output_padding", dim);
  params.groups = groups_;
  params.benchmark = benchmark;
  params.deterministic = deterministic;
  params.cudnn_enabled = cudnn_enabled;
  params.allow_tf32 = allow_tf32;

  check_shape_forward(input, weight_sizes, bias, params);

  // Expand 1d -> 2d.
  // This is only done for backends that don't natively support 1d spatial input.
  if (k == 3 && !input.is_mkldnn()) {
    // avoid accidentally going through NHWC for permuted 3d input.
    input = input.contiguous();
    params.view1d_as_2d();
    input = view4d(input);
    weight = view4d(weight);
  }

  // Select appropriate backend to use.
  auto bias_sizes_opt = bias.defined() ? 
      c10::optional<at::IntArrayRef>(bias.sizes()) : c10::nullopt;
  bool need_backward = at::GradMode::is_enabled() &&
      (input.requires_grad() || weight.requires_grad() || (bias.defined() && bias.requires_grad()));
  at::native::ConvBackend backend = 
      select_conv_backend_custom(input, weight, bias_sizes_opt, need_backward, params);
  at::MemoryFormat backend_memory_format = determine_backend_memory_format(input, weight, backend);

  LOG(INFO) << "Convolution backend: " << backend << ", memory format: " << backend_memory_format;

  // Call the backend.
  at::Tensor output;
  auto kernel_size = weight.sizes().slice(2);
  switch (backend) {
    case at::native::ConvBackend::CudaDepthwise2d:
      output = at::_conv_depthwise2d(input.contiguous(), weight, kernel_size, bias,
          params.stride, params.padding, params.dilation);
      break;
    case at::native::ConvBackend::CudaDepthwise3d:
      output = at::conv_depthwise3d(input.contiguous(), weight, kernel_size, bias,
          params.stride, params.padding, params.dilation);
      break;
    case at::native::ConvBackend::Cudnn:
      check_input_same_type_as_parameters(input, weight, bias);
      output = at::cudnn_convolution(
          input.contiguous(backend_memory_format), weight, params.padding, params.stride,
          params.dilation, params.groups, params.benchmark, params.deterministic, params.allow_tf32);
      if (bias.defined()) {
        output.add_(at::native::reshape_bias(input.dim(), bias));
      }
      break;
    case at::native::ConvBackend::CudnnTranspose:
      check_input_same_type_as_parameters(input, weight, bias);
      output = at::cudnn_convolution_transpose(
          input.contiguous(backend_memory_format), weight, params.padding, params.output_padding,
          params.stride, params.dilation, params.groups, params.benchmark, params.deterministic, params.allow_tf32);
      if (bias.defined()) {
        output.add_(at::native::reshape_bias(input.dim(), bias));
      }
      break;
    case at::native::ConvBackend::Empty:
    {
      auto weight_view = at::_unsafe_view(weight, -1);
      output = (input.size(1) == 0) ? (input.view(-1) * weight_view) : (input * weight_view[0]);
      if (bias.defined()) {
        output.add_(bias[0]);
      }
      output = output.view(calc_output_size(input, weight, params));
      break;
    }
    case at::native::ConvBackend::Slow3d:
      output = at::slow_conv3d(input, weight, kernel_size, bias, params.stride, params.padding);
      break;
    case at::native::ConvBackend::Slow2d:
    case at::native::ConvBackend::SlowDilated2d:
    case at::native::ConvBackend::SlowDilated3d:
    case at::native::ConvBackend::SlowTranspose2d:
    case at::native::ConvBackend::SlowTranspose3d:
      input = input.contiguous(backend_memory_format);
      weight = weight.contiguous(backend_memory_format);
      if (params.groups == 1) {
        output = _convolution_nogroup_backend(input, weight, bias, backend, params);
      } else {
        std::vector<at::Tensor> outputs(params.groups);
        for (const auto g : c10::irange(params.groups)) {
          auto input_g = subtensor(input, 1, params.groups, g);
          auto weight_g = subtensor(weight, 0, params.groups, g);
          auto bias_g = subtensor(bias, 0, params.groups, g);
          outputs[g] = _convolution_nogroup_backend(input_g, weight_g, bias_g, backend, params);
        }
        output = at::cat(outputs, 1);
      }
      break;
    default:
      TORCH_CHECK(false, "torch_col unsupported ConvNd parameters");
  }

  if (k == 3 && !input.is_mkldnn()) {
    output = view3d(output);
  }

  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> _convolution_backward_nogroup_backend(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::array<bool, 3> output_mask,
    const at::native::ConvBackend backend,
    const ConvParamsCustom& params) {
  auto kernel_size = weight.sizes().slice(2);
  switch(backend) {
    case at::native::ConvBackend::Slow2d:
      return at::_slow_conv2d_backward(
        grad_output, input, weight, kernel_size, params.stride, params.padding, output_mask);
    // NB: nnpack backward does not support strided convolutions; use slow impl instead
    case at::native::ConvBackend::SlowDilated2d:
      return at::native::slow_conv_dilated2d_backward_stub(
        input.device().type(),
        grad_output, input, weight, kernel_size, params.stride, params.padding, params.dilation, output_mask);
    case at::native::ConvBackend::SlowDilated3d:
      return at::native::slow_conv_dilated3d_backward_stub(
        input.device().type(),
        grad_output, input, weight, kernel_size, params.stride, params.padding, params.dilation, output_mask);
    case at::native::ConvBackend::SlowTranspose2d:
      return at::native::slow_conv_transpose2d_backward_stub(
        input.device().type(), grad_output, input, weight, kernel_size, params.stride, params.padding,
        params.output_padding, params.dilation, output_mask);
    case at::native::ConvBackend::SlowTranspose3d:
      return at::native::slow_conv_transpose3d_backward_stub(
        input.device().type(), grad_output, input, weight, kernel_size, params.stride, params.padding,
        params.output_padding, params.dilation, output_mask);
    default:
      TORCH_CHECK(false, "Unsupported conv nogroup backend encountered");
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> convolution_backward(
    const at::Tensor& grad_output_, const at::Tensor& input_, const at::Tensor& weight_,
    const at::OptionalIntArrayRef bias_sizes_opt,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding,
    int64_t groups, std::array<bool, 3> output_mask) {
  auto grad_output = grad_output_;
  auto input = input_;
  auto weight = weight_;

  auto k = weight.ndimension();
  int64_t dim = k - 2;

  TORCH_CHECK(dim > 0, "weight should have at least three dimensions");

  auto& ctx = at::globalContext();
  ConvParamsCustom params;
  params.stride = expand_param_if_needed(stride, "stride", dim);
  params.padding = expand_param_if_needed(padding, "padding", dim);
  params.dilation = expand_param_if_needed(dilation, "dilation", dim);
  params.transposed = transposed;
  params.output_padding = expand_param_if_needed(output_padding, "output_padding", dim);
  params.groups = groups;
  params.benchmark = ctx.benchmarkCuDNN();
  params.deterministic = ctx.deterministicCuDNN() || ctx.deterministicAlgorithms();
  params.cudnn_enabled = ctx.userEnabledCuDNN();
  params.allow_tf32 = ctx.allowTF32CuDNN();

  // Validate inputs.
  check_shape_backward(input, weight.sizes(), params);
  TORCH_CHECK(input.dim() == grad_output.dim(),
      "Expected input and grad_output to have the same number of dimensions, but got: ",
      input.dim(), " and ", grad_output.dim());

  // output_padding is only supported for transposed convolutions
  if (!params.transposed) {
    for (auto pad : params.output_padding) {
      TORCH_CHECK(pad == 0, "output_padding is not supported for non-transposed convolutions; got: ",
        params.output_padding);
    }
  }

  // Expand 1d -> 2d.
  // This is only done for backends that don't natively support 1d spatial input.
  if (k == 3 && !input.is_mkldnn()) {
    // avoid accidentally going through NHWC for permuted 3d input.
    input = input.contiguous();
    params.view1d_as_2d();
    grad_output = view4d(grad_output);
    input = view4d(input);
    weight = view4d(weight);
  }

  // Select appropriate backend to use.
  at::native::ConvBackend backend = select_conv_backend_custom(input, weight, bias_sizes_opt, /*need_backward=*/ true, params);
  at::MemoryFormat backend_memory_format = determine_backend_memory_format(input, weight, backend);

  LOG(INFO) << "convolution_backward backend: " << backend << ", memory format: " << backend_memory_format;

  // Call the backend.
  at::Tensor backend_grad_input, backend_grad_weight, backend_grad_bias;
  auto kernel_size = weight.sizes().slice(2);
  switch(backend) {
    case at::native::ConvBackend::CudaDepthwise2d:
    {
      std::array<bool, 2> input_weight_output_mask = {output_mask[0], output_mask[1]};
      std::tie(backend_grad_input, backend_grad_weight) =
        at::native::conv_depthwise2d_backward_stub(input.device().type(), grad_output, input,
          weight, kernel_size, params.stride, params.padding, params.dilation, input_weight_output_mask);
      break;
    }
    case at::native::ConvBackend::CudaDepthwise3d:
      TORCH_CHECK(input.ndimension() == 5);
      std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) =
        at::native::conv_depthwise3d_backward_stub(
          input.device().type(), grad_output, input, weight, kernel_size, params.stride,
          params.padding, params.dilation, output_mask);
      break;
    case at::native::ConvBackend::Cudnn:
    {
      check_input_same_type_as_parameters(input, weight);
      std::array<bool, 2> input_weight_output_mask = {output_mask[0], output_mask[1]};
      std::tie(backend_grad_input, backend_grad_weight) = at::native::cudnn_convolution_backward_stub(
          input.device().type(),
          // Only make input contiguous when it is necessary for the backwards computation
          output_mask[1] ? input.contiguous(backend_memory_format) : input,
          grad_output, weight, params.padding, params.stride,
          params.dilation, params.groups, params.benchmark, params.deterministic, params.allow_tf32,
          input_weight_output_mask);
      break;
    }
    case at::native::ConvBackend::CudnnTranspose:
    {
      check_input_same_type_as_parameters(input, weight);
      std::array<bool, 2> input_weight_output_mask = {output_mask[0], output_mask[1]};
      std::tie(backend_grad_input, backend_grad_weight) = at::native::cudnn_convolution_transpose_backward_stub(
        input.device().type(),
        // Only make input contiguous when it is necessary for the backwards computation
        output_mask[1] ? input.contiguous(backend_memory_format) : input,
        grad_output, weight, params.padding, params.output_padding,
        params.stride, params.dilation, params.groups, params.benchmark, params.deterministic, params.allow_tf32,
        input_weight_output_mask);
      break;
    }
    case at::native::ConvBackend::Empty:
      if (output_mask[0]) {
        backend_grad_input = at::zeros_like(input);
      }
      if (output_mask[1]) {
        backend_grad_weight = at::zeros_like(weight);
      }
      if (output_mask[2]) {
        backend_grad_bias = at::zeros(*bias_sizes_opt, weight.options());
      }
      break;
    case at::native::ConvBackend::Slow3d:
      // Note that no CUDA implementation of this kernel exists currently.
      LOG(FATAL) << "Slow3d is not supported for backward on CUDA";
      // we comment the following due to `at::native::slow_conv3d_backward_cpu`
      // is linked as a local symbol
      // std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) =
      //   at::native::slow_conv3d_backward_cpu(
      //       grad_output, input, weight, kernel_size,
      //       params.stride, params.padding, output_mask);
      break;
    // Handle backends that don't natively support groups > 1.
    case at::native::ConvBackend::Slow2d:
    case at::native::ConvBackend::SlowDilated2d:
    case at::native::ConvBackend::SlowDilated3d:
    case at::native::ConvBackend::SlowTranspose2d:
    case at::native::ConvBackend::SlowTranspose3d:
    {
      input = input.contiguous(backend_memory_format);
      weight = weight.contiguous(backend_memory_format);
      if (params.groups == 1) {
        std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) =
          _convolution_backward_nogroup_backend(
            grad_output, input, weight, output_mask, backend, params);
      } else {
        std::vector<at::Tensor> backend_grad_inputs(params.groups);
        std::vector<at::Tensor> backend_grad_weights(params.groups);
        std::vector<at::Tensor> backend_grad_biases(params.groups);
        for (int g = 0; g < params.groups; ++g) {
          auto grad_output_g = subtensor(grad_output, 1, params.groups, g);
          auto input_g = subtensor(input, 1, params.groups, g);
          auto weight_g = subtensor(weight, 0, params.groups, g);
          std::tie(backend_grad_inputs[g], backend_grad_weights[g], backend_grad_biases[g]) =
            _convolution_backward_nogroup_backend(
              grad_output_g, input_g, weight_g, output_mask, backend, params);
        }
        if (output_mask[0]) {
          backend_grad_input = at::cat(backend_grad_inputs, 1);
        }
        if (output_mask[1]) {
          backend_grad_weight = at::cat(backend_grad_weights, 0);
        }
        if (output_mask[2]) {
          backend_grad_bias = at::cat(backend_grad_biases, 0);
        }
      }
      break;
    }
    default:
      TORCH_CHECK(false, "torch_col unsupported ConvNd parameters");
  }

  // Convert 2D inputs back to 1D for backends that don't natively support 1D
  // spatial inputs.
  if (output_mask[0]) {
    if (k == 3 && !input.is_mkldnn()) {
      backend_grad_input = view3d(backend_grad_input);
    }
  }
  if (output_mask[1]) {
    if (k == 3 && !input.is_mkldnn()) {
      backend_grad_weight = view3d(backend_grad_weight);
    }
  }
  if (output_mask[2]) {
    if (!backend_grad_bias.defined()) {
      // Calculate bias gradients outside of the backend for those that don't support it.
      backend_grad_bias = grad_output.sum((dim == 3) ? at::IntArrayRef{0, 2, 3, 4} : at::IntArrayRef{0, 2, 3});
    }
  }

  return std::make_tuple(backend_grad_input, backend_grad_weight, backend_grad_bias);
}


}