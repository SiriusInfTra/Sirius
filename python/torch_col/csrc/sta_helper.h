#ifndef COLSERVE_STA_HELPER_H
#define COLSERVE_STA_HELPER_H

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