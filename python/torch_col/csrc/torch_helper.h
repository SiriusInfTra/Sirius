#ifndef COLSERVE_TORCH_HELPER_H
#define COLSERVE_TORCH_HELPER_H

#include <Python.h>

namespace torch_col {

void ReleaseGradFnSavedTensor(PyObject* grad_fn);


}

#endif