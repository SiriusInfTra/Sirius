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
        ctypes.CDLL(os.path.join(lib_path, 'libtorch_col_tensor.so'), ctypes.RTLD_GLOBAL)
        __setup_coltensor()

__initialize()
from ._C import *
