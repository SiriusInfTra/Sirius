  #include <Python.h>
#include <object.h>
#include <moduleobject.h>
#include <ATen/Tensor.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/object_ptr.h>

#include <common/controlling.h>

#include <torch_col/csrc/torch_allocator_plugin.h>
#include <torch_col/csrc/mem_tagging.h>
#include <torch_col/csrc/fake_engine.h>

namespace torch_col {

using namespace torch::cuda::CUDAColAllocator;

void TagModelParameterStart() {
  CUDAColAllocator::Get()->SetTrainModelAllocating(true);
}

void TagModelParameterEnd() {
  CUDAColAllocator::Get()->SetTrainModelAllocating(false);
}

void TagIntermMemory(PyObject* py_tensor) {
  at::Tensor tensor = THPVariable_Unpack(py_tensor);
  // auto *allocator = CUDAColAllocator::Get();
  // at::Storage zero_storage(
  // c10::make_intrusive<c10::StorageImpl>( c10::StorageImpl::use_byte_size_t(), 0,  c10::DataPtr{nullptr,  c10::Device(c10::DeviceType::CUDA, 0)} , allocator, true)); 
  // tensor.unsafeGetTensorImpl()->set_storage_keep_dtype(zero_storage);
  // tensor.unsafeGetTensorImpl().
  CUDAColAllocator::Get()->TagIntermMemory(tensor);
}

void ReleaseIntermMemory() {
  CUDAColAllocator::Get()->ReleaseIntermMemory();
}

void UntagIntermMemory() {
  CUDAColAllocator::Get()->UntagIntermMemory();
}

void RearrangeMemory() {
  LOG(FATAL) << "RearrangeMemory is deprecated currently";
}

void TorchColSavedVariableHooks::call_pack_hook(const at::Tensor& tensor) {
  CUDAColAllocator::Get()->TagIntermMemory(tensor);
  data_ = tensor;
  if (static_cast<ctrl::CtrlEvent>(GetColocateStub().Cmd()) == ctrl::CtrlEvent::kColocateAdjustL1) {
    pybind11::gil_scoped_acquire gil;
    throw EngineColocateAdjustL1Exception("TorchColEngine");
  }
}

at::Tensor TorchColSavedVariableHooks::call_unpack_hook() {
  if (static_cast<ctrl::CtrlEvent>(GetColocateStub().Cmd()) == ctrl::CtrlEvent::kColocateAdjustL1) {
    pybind11::gil_scoped_acquire gil;
    LOG(INFO) << "throw python exception!";
    throw EngineColocateAdjustL1Exception("TorchColEngine");
  }
  return data_;
}
}  // namespace torch_col