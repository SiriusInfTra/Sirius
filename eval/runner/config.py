import os
import numpy as np
import contextlib
import hashlib

__global_seed = None

def get_global_seed():
    global __global_seed
    if __global_seed is None:
        __global_seed = np.random.randint(2<<31)
        print(f'\x1b[33;1mWARNING: global seed is not set, set to {__global_seed}\x1b[0m')
    return __global_seed

def get_global_seed_by_hash(s:str):
    seed = get_global_seed()
    seed = seed + eval('0x' + hashlib.md5(s.encode()).hexdigest()[:4])
    # print(f'{s}, seed = {seed}, global_seed = {get_global_seed()}')
    return seed

def set_global_seed(seed):
    global __global_seed
    __global_seed = seed


def set_mps_thread_percent(percent):
    os.environ['_CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = str(percent)


def unset_mps_thread_percent():
    os.environ.pop('_CUDA_MPS_ACTIVE_THREAD_PERCENTAGE', None)


@contextlib.contextmanager
def mps_thread_percent(percent, skip=False):
    if percent is None or skip:
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