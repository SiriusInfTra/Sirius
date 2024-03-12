#include <ATen/Tensor.h>
#include <torch/csrc/autograd/python_variable.h>

#include "mem_tagging.h"
#include "cuda_allocator_plugin.h"

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
  auto storage = tensor.unsafeGetTensorImpl()->storage();
  CUDAColAllocator::Get()->TagIntermMemory(storage.data(), storage.nbytes(), storage.allocator());

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



}