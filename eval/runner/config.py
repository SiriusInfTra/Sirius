import numpy as np

__global_seed = None

def get_global_seed():
    global __global_seed
    if __global_seed is None:
        __global_seed = np.random.randint(2<<31)
    return __global_seed

def set_global_seed(seed):
    global __global_seed
    __global_seed = seed