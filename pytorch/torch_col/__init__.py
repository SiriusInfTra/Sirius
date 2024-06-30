__use_shared_tensor = 0
__release_interm_memory_by_grad_fn = 0
__release_interm_memory_by_tagging = 1

__use_fbward_hook = 1

def use_shared_tensor():
    global __use_shared_tensor
    return __use_shared_tensor

def release_interm_memory_v1():
    global __release_interm_memory_by_grad_fn
    return __release_interm_memory_by_grad_fn

def release_interm_memory_v2():
    global __release_interm_memory_by_tagging
    return __release_interm_memory_by_tagging

def disable_release_interm_memory(): # used for eval
    global __release_interm_memory_by_grad_fn
    global __release_interm_memory_by_tagging
    __release_interm_memory_by_grad_fn = 0
    __release_interm_memory_by_tagging = 0

def disable_fbward_hook():
    global __use_fbward_hook
    __use_fbward_hook = 0

def use_fbward_hook():
    global __use_fbward_hook
    return __use_fbward_hook

from ._C import *

from .util import MemoryPool, TrainMode, EventManager
from .accumulate import GradAccumulator

from .dataset import CustomeDynamicBatchDataset
from .hook import register_saved_tensor_hook, get_hook, HookMode, DummyHook,\
      SwitchHook, SwitchL1Exception, ColocateHook, ColocateAdjustL1Exception
from .debug_server import DebugServer


class EngineColocateAdjustL1Exception(Exception):
    pass

def __setup_torch_col():
    import os, sys
    if os.environ.get('COL_USE_SHARED_TENSOR', '0') == '1':
        global __use_shared_tensor
        __use_shared_tensor = 1
    torch_col_init(use_shared_tensor())

    # TorchColConfig.InitConfig(use_shared_tensor())
    # CUDAColAllocator.Init()
    # if use_shared_tensor():
    #     CUDAColAllocator.Get().init(0)
    #     CUDAColAllocator.SetCurrentAllocator()



__setup_torch_col()

print('torch_col init done') 