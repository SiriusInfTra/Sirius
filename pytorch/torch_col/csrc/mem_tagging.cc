#include <ATen/Tensor.h>
#include <torch/csrc/autograd/python_variable.h>

#include "common/controlling.h"
#include "mem_tagging.h"
#include "cuda_allocator_plugin.h"
#include "torch_col/csrc/fake_engine.h"

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

  // auto col_tensor = GetColTensorImpl(tensor);
  // DLOG(INFO) << "[TORCH_COL STA] " << "tas as saved tensor " << col_tensor->Handle() << " " << col_tensor->CTensor().MData()->addr;
  // colserve::sta::TensorPool::Get()->AddTrainIntermediateTensor(col_tensor->Handle());
}

void ReleaseIntermMemory() {
  // colserve::sta::TensorPool::Get()->ReleaseTrainIntermediateTensorMemory();
  CUDAColAllocator::Get()->ReleaseIntermMemory();
}

void UntagIntermMemory() {
    CUDAColAllocator::Get()->UntagIntermMemory();
  // colserve::sta::TensorPool::Get()->ClearTrainIntermediateTensor();
}

void RearrangeMemory() {
  // colserve::sta::TensorPool::Get()->RearrangeTrainMemory();
}



void TorchColSavedVariableHooks::call_pack_hook(const at::Tensor& tensor) {
  CUDAColAllocator::Get()->TagIntermMemory(tensor);
  data_ = tensor;
  if (static_cast<ctrl::CtrlEvent>(GetColocateStub().Cmd()) == ctrl::CtrlEvent::kColocateAdjustL1) {
    
  }

}

at::Tensor TorchColSavedVariableHooks::call_unpack_hook() {
  return data_;
}
}  // namespace torch_col