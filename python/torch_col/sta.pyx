# cython: c_string_type=unicode, c_string_encoding=utf8
from cpython.ref cimport PyObject


cdef extern from "<csrc/sta_helper.h>" namespace "torch_col":
    cdef void TagModelParameterStart()
    cdef void TagModelParameterEnd()
    cdef void TagAsIntermediateTensor(PyObject* obj)
    cdef void ReleaseIntermediateTensorMemory()
    cdef void ClearIntermediateTensor()
    cdef void RearrangeMemory()
    

def tag_model_start():
    TagModelParameterStart()


def tag_model_end():
    TagModelParameterEnd()


def tag_as_saved_tensor(tensor):
    cdef PyObject* obj = <PyObject*> tensor
    TagAsIntermediateTensor(obj)


def release_saved_tensor_memory():
    ReleaseIntermediateTensorMemory()


def clear_saved_tensor():
    ClearIntermediateTensor()

def rearrange_memory():
    RearrangeMemory()
     