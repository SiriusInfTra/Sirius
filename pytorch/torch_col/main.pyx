# cython: c_string_type=unicode, c_string_encoding=utf8

include "./memory.pxi"
include "./ctrl_stub.pxi"

cdef extern from "<csrc/config.h>" namespace "torch_col":
    cpdef void ConfigTorchCol(bint)
