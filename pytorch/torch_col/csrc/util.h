#ifndef TORCH_COL_UTIL_H
#define TORCH_COL_UTIL_H

#include <c10/core/TensorImpl.h>
#include <pybind11/pytypes.h>
#include <Python.h>
#include <chrono>
#include <string>
#include <optional>


namespace torch_col {
inline auto get_unix_timestamp() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
}

inline auto get_unix_timestamp_us() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count();
}

void ReleaseGradFnSavedTensor(PyObject* grad_fn);
void ReleaseUnderlyingStorage(PyObject* tensor);

void DumpMempoolFreeList(std::string filename);
void DumpMempoolBlockList(std::string filename);


class TensorWeakRef {
 public:
  TensorWeakRef(PyObject* tensor);
  size_t Nbytes() const;
  size_t StorageNbytes() const;
  void* DataPtr() const;

 private:
  std::optional<c10::weak_intrusive_ptr<at::TensorImpl>> tensor_weak_ref_;
};



}

#endif