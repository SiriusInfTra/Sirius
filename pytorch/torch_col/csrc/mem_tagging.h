#ifndef TORCH_COL_MEM_TAGGING_H
#define TORCH_COL_MEM_TAGGING_H

#include <Python.h>

namespace torch_col {

void TagModelParameterStart();
void TagModelParameterEnd();
void TagIntermMemory(PyObject* tensor);
void ReleaseIntermMemory();
void UntagIntermMemory();
void RearrangeMemory();

}

#endif