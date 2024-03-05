#ifndef TORCH_COL_UTIL_H
#define TORCH_COL_UTIL_H

#include <chrono>
#include <string>
#include <Python.h>

namespace torch_col {
inline auto get_unix_timestamp() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

inline auto get_unix_timestamp_us() {
  return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

void ReleaseGradFnSavedTensor(PyObject* grad_fn);
void ReleaseUnderlyingStorage(PyObject* tensor);

void DumpMempoolFreeList(std::string filename);
void DumpMempoolBlockList(std::string filename);


}

#endif