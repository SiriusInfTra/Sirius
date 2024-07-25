from ._C import *

from .util import MemoryPool, TrainMode, EventManager
from .accumulate import GradAccumulator

from .dataset import CustomeDynamicBatchDataset
from .hook import register_saved_tensor_hook, get_hook, HookMode, DummyHook,\
      SwitchHook, SwitchL1Exception, ColocateHook, ColocateAdjustL1Exception
from .debug_server import DebugServer


class EngineColocateAdjustL1Exception(Exception):
    pass


# torch_col_init()