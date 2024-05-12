import os
import numpy as np
import contextlib

__global_seed = None

def get_global_seed():
    global __global_seed
    if __global_seed is None:
        __global_seed = np.random.randint(2<<31)
    return __global_seed

def set_global_seed(seed):
    global __global_seed
    __global_seed = seed


def set_mps_thread_percent(percent):
    os.environ['_CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = str(percent)


def unset_mps_thread_percent():
    os.environ.pop('_CUDA_MPS_ACTIVE_THREAD_PERCENTAGE', None)


@contextlib.contextmanager
def mps_thread_percent(percent, skip=False):
    if skip:
        yield
        return
    else:
        set_mps_thread_percent(percent)
        yield
        unset_mps_thread_percent()


@contextlib.contextmanager
def um_mps(percent):
    set_mps_thread_percent(percent)
    os.environ['TORCH_UNIFIED_MEMORY'] = "1"
    os.environ['STA_RAW_ALLOC_UNIFIED_MEMORY'] = "1"
    yield
    unset_mps_thread_percent()
    os.environ.pop('TORCH_UNIFIED_MEMORY', None)
    os.environ.pop('STA_RAW_ALLOC_UNIFIED_MEMORY', None)