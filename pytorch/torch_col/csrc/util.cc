#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/python_cpp_function.h>

#include <common/mempool_sampler.h>
#include <common/util.h>

#include <string>


namespace torch_col {

void ReleaseGradFnSavedTensor(PyObject* py_grad_fn) {
  auto grad_fn = (::torch::autograd::THPCppFunction*)py_grad_fn;
  auto function = static_cast<::torch::autograd::Node*>(grad_fn->cdata.get());
  function->will_release_variables();
  function->release_variables();
}

void DumpMempoolFreeList(std::string filename) {
  colserve::sta::DumpMempoolFreeList(filename);
}
void DumpMempoolBlockList(std::string filename) {
    colserve::sta::DumpMempoolBlockList(filename);
}


}