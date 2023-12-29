#include <torch/csrc/autograd/python_variable.h>

#include "tensor_impl.h"
#include "sta_helper.h"

namespace torch_col {
namespace {
inline ColTensorImpl* GetColTensorImpl(const at::Tensor& tensor) {
auto impl = dynamic_cast<ColTensorImpl*>(tensor.unsafeGetTensorImpl());
CHECK(impl) << "input tensor is not a ColTensor, got *impl " << tensor.unsafeGetTensorImpl() 
            << " data_ptr " << tensor.data_ptr() << " " << tensor.toString()
            << " , device " << tensor.device() << " " << tensor.key_set();
return impl;
}
}

void TagModelParameterStart() {
  colserve::sta::TensorPool::Get()->SetTrainModelAllocating(true);
}

void TagModelParameterEnd() {
  colserve::sta::TensorPool::Get()->SetTrainModelAllocating(false);
}

void TagAsIntermediateTensor(PyObject* py_tensor) {
  auto tensor = THPVariable_Unpack(py_tensor);
  auto col_tensor = GetColTensorImpl(tensor);
  DLOG(INFO) << "[TORCH_COL STA] " << "tas as saved tensor " << col_tensor->Handle() << " " << col_tensor->CTensor().MData()->addr;
  colserve::sta::TensorPool::Get()->AddTrainIntermediateTensor(col_tensor->Handle());
}

void ReleaseIntermediateTensorMemory() {
  colserve::sta::TensorPool::Get()->ReleaseTrainIntermediateTensorMemory();
}

void ClearIntermediateTensor() {
  colserve::sta::TensorPool::Get()->ClearTrainIntermediateTensor();
}

void RearrangeMemory() {
  colserve::sta::TensorPool::Get()->RearrangeTrainMemory();
}

// PyObject* TagAsIntermediateTensor(PyObject* self, PyObject* noargs) {
//   auto tensor = THPVariable_Unpack(self);
//   auto col_tensor = GetColTensorImpl(tensor);
//   colserve::sta::TensorPool::Get()->AddTrainIntermediateTensor(col_tensor->Handle());
//   Py_RETURN_NONE;
// }

// PyObject* ReleaseIntermediateTensorMemory(PyObject* unused, PyObject* noargs) {
//   colserve::sta::TensorPool::Get()->ReleaseTrainIntermediateTensorMemory();
//   Py_RETURN_NONE;
// }

// PyObject* ClearIntermediateTensor(PyObject* unused, PyObject* noargs) {
//   colserve::sta::TensorPool::Get()->ClearTrainIntermediateTensor();
//   Py_RETURN_NONE;
// }

// ---------------------- Python Module ----------------------

// static struct PyMethodDef sta_methods[] = {
//   {"tag_as_saved_tensor", TagAsIntermediateTensor, METH_O, nullptr},
//   {"release_saved_tensor", ReleaseIntermediateTensorMemory, METH_NOARGS, nullptr},
//   {"clear_saved_tensor", ClearIntermediateTensor, METH_NOARGS, nullptr},
//   {nullptr, nullptr, 0, nullptr}
// };

// static struct PyModuleDef torch_col_sta = {
//   PyModuleDef_HEAD_INIT,
//   "torch_col.sta", // name of module
//   "extra shared tensor allocator", // module documentation
//   -1, // size of per-interpreter state of the module
//   sta_methods
// };

// PyInit_torch_col_sta(void) {
//   LOG(INFO) << "initialize torch.sta module";
//   return PyModule_Create(&torch_col_sta);
// }

// struct PyModuleInitializer_ColTensorSTA {
//   PyModuleInitializer_ColTensorSTA() {
//     PyModule_Create(&torch_col_sta);
//   }
// };


}