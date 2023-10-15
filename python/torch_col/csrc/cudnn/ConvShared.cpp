#include "ConvShared.h"

namespace at { namespace native {

// ---------------------------------------------------------------------
//
// ConvolutionParams
//
// ---------------------------------------------------------------------

std::ostream& operator<<(std::ostream & out, const at::native::ConvolutionParams& params) {
  out << "ConvolutionParams \n"
    << "    memory_format = " << params.memory_format << "\n"
    << "    data_type = " << at::native::cudnnTypeToString(params.dataType) << "\n"
    << "    padding = " << at::ArrayRef<int>{params.padding} << "\n"
    << "    stride = " << at::ArrayRef<int>{params.stride} << "\n"
    << "    dilation = " << at::ArrayRef<int>{params.dilation} << "\n"
    << "    groups = " << params.groups << "\n"
    << "    deterministic = " << (params.deterministic ? "true" : "false") << "\n"
    << "    allow_tf32 = " << (params.allow_tf32 ? "true" : "false") << "\n";

  return out;
}


std::string repro_from_args(const at::native::ConvolutionParams& params) {
  auto pybool = [](bool b) -> const char* { return b ? "True" : "False"; };
  std::string partial_dtype;
  switch (params.dataType) {
    case CUDNN_DATA_FLOAT: partial_dtype = "float"; break;
    case CUDNN_DATA_DOUBLE: partial_dtype = "double"; break;
    case CUDNN_DATA_HALF: partial_dtype = "half"; break;
    default: partial_dtype = "unsupported";
  }
  const std::string full_dtype = "torch." + partial_dtype;
  const int out_channels = params.weight_size[0];
  const int in_channels = params.weight_size[1] * params.groups;
  const size_t dim = params.input_dim;
  const std::string channels_last_xd = dim == 4 ? "channels_last" : "channels_last_3d";
  const std::string to_channels_last =
    ((params.memory_format == at::MemoryFormat::ChannelsLast) || (params.memory_format == at::MemoryFormat::ChannelsLast3d)) \
    ? ".to(memory_format=torch." + channels_last_xd + ")" : "";

  std::ostringstream ss;
  ss << "You can try to repro this exception using the following code snippet. ";
  ss << "If that doesn't trigger the error, please include your original repro script when reporting this issue.\n\n";
  ss << "import torch\n";
  ss << "torch.backends.cuda.matmul.allow_tf32 = " << pybool(at::globalContext().allowTF32CuBLAS()) << "\n";
  ss << "torch.backends.cudnn.benchmark = " << pybool(at::globalContext().benchmarkCuDNN()) << "\n";
  ss << "torch.backends.cudnn.deterministic = " << pybool(params.deterministic) << "\n";
  ss << "torch.backends.cudnn.allow_tf32 = " << pybool(params.allow_tf32) << "\n";
  ss << "data = torch.randn(" << at::ArrayRef<int>(params.input_size, dim) << ", dtype=" << full_dtype << ", ";
  ss <<   "device='cuda', requires_grad=True)" << to_channels_last << "\n";
  ss << "net = torch.nn.Conv" << dim-2 << "d(" << in_channels << ", " << out_channels << ", ";
  ss <<   "kernel_size=" << at::ArrayRef<int>(&params.weight_size[2], dim - 2) << ", ";
  ss <<   "padding=" << at::ArrayRef<int>(params.padding, dim-2) << ", ";
  ss <<   "stride=" << at::ArrayRef<int>(params.stride, dim-2) << ", ";
  ss <<   "dilation=" << at::ArrayRef<int>(params.dilation, dim-2) << ", ";
  ss <<   "groups=" << params.groups << ")\n";
  ss << "net = net.cuda()." << partial_dtype << "()" << to_channels_last << "\n";
  ss << "out = net(data)\n";
  ss << "out.backward(torch.randn_like(out))\n";
  ss << "torch.cuda.synchronize()\n\n";

  return ss.str();
}

// NB: This can't be a constructor, because then ConvolutionParams
// would not be a POD anymore.
// TODO: Use TensorGeometry here instead of the entire Tensor, which we
// don't actually need.  (OTOH: We can always pass in
// grad_input/grad_output, so this is not very pressing)
void setConvolutionParams(
    at::native::ConvolutionParams* params,
    const at::Tensor& input, const at::Tensor& weight,
    at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation,
    int64_t groups, bool deterministic, bool allow_tf32, at::MemoryFormat memory_format) {

  cudnnDataType_t dataType = at::native::getCudnnDataType(input);
  memset(params, 0, sizeof(at::native::ConvolutionParams));
  params->device_id = at::cuda::current_device();
  params->dataType = dataType;
  // ASSERT(weight.dim() == input.dim())
  params->input_dim = input.dim();
  params->memory_format = memory_format;
  for (int i = 0; i != params->input_dim; ++i) {
    params->input_size[i] = (int) input.sizes()[i];
    params->weight_size[i] = (int) weight.sizes()[i];
  }
  // ASSERT(padding.size() == stride.size())
  // ASSERT(padding.size() == dilation.size())
  for (size_t i = 0; i != padding.size(); ++i) {
    params->padding[i] = padding[i];
    params->stride[i] = stride[i];
    params->dilation[i] = dilation[i];
  }
  // In principle, we shouldn't parametrize by groups for legacy
  // CuDNN, but it doesn't seem worth the effort to actually do this.
  params->groups = groups;
  params->deterministic = deterministic;
  params->allow_tf32 = allow_tf32;
}


}}