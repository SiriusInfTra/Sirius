import os
import numpy as np
import contextlib
import hashlib
import pathlib


# used for docker
HOST_DOCKER_RUN_DIR = os.environ.get('HOST_DOCKER_RUN_DIR', None)
DOCKER_TRITON_MODEL_WKSP = os.environ.get('DOCKER_TRITON_MODEL_WKSP', 'triton-model-wksp') 
DOCKER_GPU_COL_LOG_DIR = os.environ.get('DOCKER_GPU_COL_LOG_DIR', 'gpu-col-docker-log')
DOCKER_MPS_PIPE_DIRECTORY = os.environ.get('DOCKER_MPS_PIPE_DIRECTORY', '/dev/shm/')


class RunnerConfig:
    _global_seed = None
    _binary_dir = 'build'
    _multi_gpu_scale_up_workload = True

    # install path of tensorrt backend with unified-memory
    # env: TENSORRT_BACKEND_UNIFIED_MEMORY_PATH
    _tensorrt_backend_unified_memory_path = \
        pathlib.Path(os.path.abspath(__file__)).parent.parent / 'triton' / 'tensorrt_um' / 'install'
    

__global_seed = None
__binary_dir = 'build'

def set_binary_dir(binary_dir):
    global __binary_dir
    __binary_dir = binary_dir


def get_binary_dir():
    global __binary_dir
    return __binary_dir


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


def get_host_name():
    return os.uname().nodename


def is_meepo5():
    return get_host_name() == 'meepo5' and not pathlib.Path('/.dockerenv').exists()


def is_inside_docker():
    return pathlib.Path('/.dockerenv').exists()


def get_tensorrt_backend_unified_memory_path():
    if 'TENSORRT_BACKEND_UNIFIED_MEMORY_PATH' in os.environ:
            return os.environ['TENSORRT_BACKEND_UNIFIED_MEMORY_PATH']

    if (RunnerConfig._tensorrt_backend_unified_memory_path is None or 
        not os.path.exists(RunnerConfig._tensorrt_backend_unified_memory_path)
    ):    
        raise RuntimeError(f"path to triton tensorrt backend with unified-memory "
                           f"({RunnerConfig._tensorrt_backend_unified_memory_path})"
                           f" not found, or set env var `TENSORRT_BACKEND_UNIFIED_MEMORY_PATH`"
                           f" to use other path")
    else:
        if isinstance(RunnerConfig._tensorrt_backend_unified_memory_path, str):
            tensorrt_backend_path = (
                pathlib.Path(RunnerConfig._tensorrt_backend_unified_memory_path) 
                / 'backends' / 'tensorrt' / 'libtriton_tensorrt.so'
            )
        else:
            tensorrt_backend_path = (
                RunnerConfig._tensorrt_backend_unified_memory_path 
                / 'backends' / 'tensorrt' / 'libtriton_tensorrt.so'
            )
        if not tensorrt_backend_path.exists():
            raise RuntimeError(f"libtriton_tensorrt.so not found at {tensorrt_backend_path}")
        return RunnerConfig._tensorrt_backend_unified_memory_path
    

def set_tensorrt_backend_unified_memory_path(path):
    RunnerConfig._tensorrt_backend_unified_memory_path = path


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