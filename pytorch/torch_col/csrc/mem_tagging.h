#ifndef TORCH_COL_MEM_TAGGING_H
#define TORCH_COL_MEM_TAGGING_H

#include <Python.h>
#include <ATen/core/TensorBody.h>
#include <Python.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/saved_variable_hooks.h>

namespace torch_col {

struct EngineColocateAdjustL1Exception : public torch::PyTorchError {
  EngineColocateAdjustL1Exception(const std::string &msg) {
    this->msg = msg;
  }
  PyObject* python_type() override {
    auto *torch_col = PyImport_ImportModule("torch_col");
    auto *py_exception = PyObject_GetAttrString(torch_col, "EngineColocateAdjustL1Exception");
    return py_exception;
  }
};

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