#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/python_cpp_function.h>

#include "torch_helper.h"

namespace torch_col {

void ReleaseGradFnSavedTensor(PyObject* py_grad_fn) {
  auto grad_fn = (::torch::autograd::THPCppFunction*)py_grad_fn;
  auto function = static_cast<::torch::autograd::Node*>(grad_fn->cdata.get());
  function->will_release_variables();
  function->release_variables();
}

}