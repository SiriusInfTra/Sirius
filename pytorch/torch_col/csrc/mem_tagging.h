#ifndef TORCH_COL_MEM_TAGGING_H
#define TORCH_COL_MEM_TAGGING_H

#include <ATen/core/TensorBody.h>
#include <Python.h>
#include <torch/csrc/autograd/saved_variable_hooks.h>

namespace torch_col {

class TorchColSavedVariableHooks : public torch::autograd::SavedVariableHooks {
private:
  at::Tensor data_;
public:
  void call_pack_hook(const at::Tensor& tensor) override;
  at::Tensor call_unpack_hook() override;
};

void TagModelParameterStart();
void TagModelParameterEnd();
void TagIntermMemory(PyObject* tensor);
void ReleaseIntermMemory();
void UntagIntermMemory();
void RearrangeMemory();

}

#endif