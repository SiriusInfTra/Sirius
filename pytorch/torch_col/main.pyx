# cython: c_string_type=unicode, c_string_encoding=utf8
# distutils: language = c++
include "./memory.pxi"
include "./ctrl_stub.pxi"
include "./sm.pxi"

from libcpp.string cimport string
from enum import Enum


cdef extern from "<csrc/config.h>" namespace "torch_col":
    cdef cppclass TorchColConfig:
        @staticmethod
        void InitConfig(bint)

        @staticmethod
        bint EnableDynamicSmPartition()

        @staticmethod
        bint EnableXsched()

        @staticmethod
        string GetHookMode()


def enable_dynamic_sm_partition():
    return TorchColConfig.EnableDynamicSmPartition()

def enable_xsched():
    return TorchColConfig.EnableXsched()

class HookMode(Enum):
    NONE = 'none'
    SYNC = 'sync'
    # XSCHED_ASYNC_SIGNAL = 'xsched-async-signal'  
    XSCHED_SYNC = 'xsched-sync'
    XSCHED_SYNC2 = 'xsched-sync2'

    def use_xsched(self):
        return self in {HookMode.XSCHED_SYNC, HookMode.XSCHED_SYNC2}

def get_hook_mode():
    # return TorchColConfig.GetHookMode()
    cdef hook_mode_cstr = TorchColConfig.GetHookMode()
    for hook_mode in HookMode:
        if hook_mode.value == hook_mode_cstr:
            return hook_mode
    raise Exception(f"Invalid hook mode: {hook_mode_cstr}")


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





