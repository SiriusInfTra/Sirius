#ifndef COLSERVE_OVERRIDE_OPS_H
#define COLSERVE_OVERRIDE_OPS_H

#include <ATen/Tensor.h>

namespace torch_col {

at::Tensor& nonzero_out_cuda(const at::Tensor& self, at::Tensor& out);

bool _has_compatible_shallow_copy_type(const at::Tensor & self, const at::Tensor & from);

}

#endif