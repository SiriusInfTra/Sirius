import os, sys
import pynvml

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

os.environ['GLOG_logtostderr'] = "1"
if 'https_proxy' in os.environ:
    del os.environ['https_proxy']

print('CUDA_VISIBLE_DEVICES={}, CUDA_MPS_PIPE_DIRECTORY={}'.format(
    os.environ['CUDA_VISIBLE_DEVICES'], os.environ['CUDA_MPS_PIPE_DIRECTORY']))

from runner import *
from workload import *