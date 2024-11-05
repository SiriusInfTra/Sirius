import os
import subprocess
import torch
from torchvision import models
import onnx
from tvm.driver import tvmc
import tvm
from tvm import relay
from tvm.relay.backend import Executor
import tempfile
import tarfile
import pathlib
import timm

import transformers
from transformers import ViTForImageClassification

platform = 'v100'

batch_size = 1

model_name = 'efficientnet_v2_s'
efficientnet_v2_s = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT).eval()


torch.onnx.export(efficientnet_v2_s, torch.rand(batch_size, 3, 224, 224), 
                  f"{model_name}.onnx",verbose=False,
                  input_names=["input0"], output_names=["output0"], 
                  export_params=True)


if 'COLSYS_TENSORRT_HOME' in os.environ:
    tensorrt_home = os.environ['COLSYS_TENSORRT_HOME']
    amd64_home = f'{tensorrt_home}/targets/x86_64-linux-gnu'
    envs = os.environ.copy()
    if 'LD_LIBRARY_PATH' in envs:
        envs['LD_LIBRARY_PATH'] = f'{amd64_home}/lib:{envs["LD_LIBRARY_PATH"]}'
    else:
        envs['LD_LIBRARY_PATH'] = f'{amd64_home}/lib'
    subprocess.run([f'{amd64_home}/bin/trtexec', f'--onnx={model_name}.onnx', f'--saveEngine={model_name}.plan'], 
        env=envs)