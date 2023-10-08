import os, sys
import ctypes


lib_path = os.path.join(os.path.dirname(__file__), 'lib')

# ctypes.CDLL(os.path.join(lib_path, 'libsta.so'), ctypes.RTLD_GLOBAL)
ctypes.CDLL(os.path.join(lib_path, 'libtorch_col.so'), ctypes.RTLD_GLOBAL)

from ._C import *