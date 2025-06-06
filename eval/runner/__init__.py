import os, sys
import pynvml
import pathlib

GPU_UUIDs = []
pynvml.nvmlInit()
for i in range(pynvml.nvmlDeviceGetCount()):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    uuid_as_str = pynvml.nvmlDeviceGetUUID(handle)
    if not isinstance(uuid_as_str, str):
        uuid_as_str = uuid_as_str.decode()
    GPU_UUIDs.append(uuid_as_str)
print(GPU_UUIDs)

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
    CUDA_VISIBLE_DEVICES_UUID = []
    for id in CUDA_VISIBLE_DEVICES.split(","):
        try:
            CUDA_VISIBLE_DEVICES_UUID.append(GPU_UUIDs[int(id.strip())])
        except:
            CUDA_VISIBLE_DEVICES_UUID.append(id.strip())
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(CUDA_VISIBLE_DEVICES_UUID)
    if not pathlib.Path('/.dockerenv').exists():
        os.environ['CUDA_MPS_PIPE_DIRECTORY'] = os.path.join(os.environ['HOME'], f'cuda_mps_{CUDA_VISIBLE_DEVICES_UUID[0]}')
    else:
        # use shared memory inside docker
        from .config import DOCKER_MPS_PIPE_DIRECTORY
        os.environ['CUDA_MPS_PIPE_DIRECTORY'] = os.path.join(
            DOCKER_MPS_PIPE_DIRECTORY, f'cuda_mps_{CUDA_VISIBLE_DEVICES_UUID[0]}')

os.environ['GLOG_logtostderr'] = "1"
if 'https_proxy' in os.environ:
    del os.environ['https_proxy']

print('CUDA_VISIBLE_DEVICES={}, CUDA_MPS_PIPE_DIRECTORY={}'.format(
    os.environ['CUDA_VISIBLE_DEVICES'], os.environ['CUDA_MPS_PIPE_DIRECTORY']))


from .config import (
    get_global_seed, get_global_seed_by_hash, set_global_seed, 
    set_mps_thread_percent, unset_mps_thread_percent, mps_thread_percent, um_mps, 
    set_binary_dir, get_binary_dir,
    RunnerConfig
)
from .runner import *
from .hyper_workload import *
from .parser import LogParser, TestUnit


def get_num_gpu():
    cuda_device_env = os.environ['CUDA_VISIBLE_DEVICES']
    return len(cuda_device_env.split(','))


def is_multi_gpu():
    return get_num_gpu() > 1


def is_four_gpu():
    return get_num_gpu() == 4


def scale_up_by_num_gpu(value):
    if RunnerConfig._multi_gpu_scale_up_workload:
        return value * get_num_gpu()
    else:
        return value
