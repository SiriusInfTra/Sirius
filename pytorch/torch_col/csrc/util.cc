#include <common/util.h>
#include <torch_col/csrc/util.h>
#include <torch_col/csrc/config.h>

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/python_cpp_function.h>
#include <torch/csrc/autograd/python_variable.h>

#include <string>


namespace torch_col {

void CallGLOG_INFO(const char* log_str, const char* file, int line) {
  google::LogMessage(file, line).stream()
      << "[Rank " << TorchColConfig::GetTrainRank() << "] " << log_str;
}

void CallGLOG_DINFO(const char* str, const char* file, int line) {
#ifndef NDEBUG
  CallGLOG_INFO(str, file, line);
#endif
}

void ReleaseGradFnSavedTensor(PyObject* py_grad_fn) {
  auto grad_fn = (::torch::autograd::THPCppFunction*)py_grad_fn;
  auto function = static_cast<::torch::autograd::Node*>(grad_fn->cdata.get());
  function->will_release_variables();
  function->release_variables();
}

void ReleaseUnderlyingStorage(PyObject* py_tensor) {
  at::Tensor tensor = THPVariable_Unpack(py_tensor);
  auto storage = tensor.unsafeGetTensorImpl()->storage();
  storage.unsafeGetStorageImpl()->reset();
}

void DumpMempoolFreeList(std::string filename) {
  LOG(FATAL) << "removed, to be re-impl"; 
}
void DumpMempoolBlockList(std::string filename) {
  LOG(FATAL) << "removed, to be re-impl"; 
}

TensorWeakRef::TensorWeakRef(PyObject* py_tensor) {
  at::Tensor tensor = THPVariable_Unpack(py_tensor);
  tensor_weak_ref_.emplace(tensor.getIntrusivePtr());
}

size_t TensorWeakRef::Nbytes() const {
  if (tensor_weak_ref_.has_value()) {
    if (auto tensor = tensor_weak_ref_.value().lock()) {
      return tensor->numel() * tensor->itemsize();
    }
  }
  return 0;
}

size_t TensorWeakRef::StorageNbytes() const {
  if (tensor_weak_ref_.has_value()) {
    if (auto tensor = tensor_weak_ref_.value().lock()) {
      return tensor->storage().nbytes();
    }
  }
  return 0;
}

const void* TensorWeakRef::DataPtr() const {
  if (tensor_weak_ref_.has_value()) {
    if (auto tensor = tensor_weak_ref_.value().lock()) {
      return tensor->data();
    }
  }
  return nullptr;
}

}