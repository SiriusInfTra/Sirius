#ifndef TORCH_COL_MEM_TAGGING_H
#define TORCH_COL_MEM_TAGGING_H

#include <Python.h>

namespace torch_col {

void TagModelParameterStart();
void TagModelParameterEnd();
void TagAsIntermediateTensor(PyObject* tensor);
void ReleaseIntermediateTensorMemory();
void ClearIntermediateTensor();
void RearrangeMemory();

}

#endif