# cython: c_string_type=unicode, c_string_encoding=utf8
# distutils: language = c++
include "./memory.pxi"
include "./ctrl_stub.pxi"
include "./sm.pxi"

cdef extern from "<csrc/config.h>" namespace "torch_col":
    cdef cppclass TorchColConfig:
        @staticmethod
        void InitConfig(bint)

        @staticmethod
        bint GetDynamicSmPartition()


cdef extern from "<csrc/xsched.h>" namespace "torch_col":
    cpdef void InitSMPartition()


def torch_col_init(use_shared_tensor: bool):
    import sys

    TorchColConfig.InitConfig(use_shared_tensor)    
    CUDAColAllocator.Init()
    if use_shared_tensor:
        CUDAColAllocator.Get().init(0)
        CUDAColAllocator.SetCurrentAllocator()
        # print("CUDAColAllocator initialized", file=sys.stderr, flush=True)





