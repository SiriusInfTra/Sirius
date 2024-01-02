__use_shared_tensor = 0
__release_saved_tensor_by_grad_fn = 0
__release_saved_tensor_by_tagging = 1

def use_shared_tensor():
    global __use_shared_tensor
    return __use_shared_tensor

def release_saved_tensor_v1():
    global __release_saved_tensor_by_grad_fn
    return __release_saved_tensor_by_grad_fn

def release_saved_tensor_v2():
    global __release_saved_tensor_by_tagging
    return __release_saved_tensor_by_tagging

def __setup_coltensor():
    import torch
    torch.Tensor.is_cuda = property(lambda self: self.device.type == 'cuda')

def __initialize():
    import os, sys
    import ctypes

    lib_path = os.path.join(os.path.dirname(__file__), 'lib')

    # ctypes.CDLL(os.path.join(lib_path, 'libsta.so'), ctypes.RTLD_GLOBAL)
    ctypes.CDLL(os.path.join(lib_path, 'libtorch_col.so'), ctypes.RTLD_GLOBAL)

    if os.environ.get('USE_SHARED_TENSOR', '0') == '1':
        global __use_shared_tensor
        __use_shared_tensor = 1
    #     ctypes.CDLL(os.path.join(lib_path, 'libtorch_col_tensor.so'), ctypes.RTLD_GLOBAL)
    #     __setup_coltensor()

__initialize()
from ._C import *

if use_shared_tensor():
    import os
    import ctypes
    lib_path = os.path.join(os.path.dirname(__file__), 'lib')
    ctypes.CDLL(os.path.join(lib_path, 'libtorch_col_tensor.so'), ctypes.RTLD_GLOBAL)
    __setup_coltensor()
    from ._C_sta import *


from .util import MemoryPool, TrainMode, EventManager
from .dataset import CustomeDynamicBatchDataset
from .hook import register_saved_tensor_hook, get_hook, HookMode, DummyHook,\
      SwitchHook, SwitchL1Exception, ColocateHook, ColocateAdjustL1Exception
from .debug_server import DebugServer
    