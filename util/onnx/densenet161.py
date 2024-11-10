import torch
from torchvision import models
import os
import subprocess


model_name = "densenet161"
batch_size = 4
densenet161 = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT).eval()
torch.onnx.export(densenet161, torch.rand(batch_size, 3, 224, 224), 
                f"{model_name}.onnx", verbose=True,
                input_names=["input0"], output_names=["output0"], export_params=True)

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