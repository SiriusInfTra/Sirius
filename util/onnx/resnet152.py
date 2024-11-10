import subprocess
import torch
from torchvision import models
import tempfile

batch_size = 4
model_store = "server/models"
model_name = "resnet152"

tmp_dir = tempfile.gettempdir()

resnet152 = models.resnet152(weights=models.ResNet152_Weights.DEFAULT).eval()
torch.onnx.export(resnet152, torch.rand(batch_size, 3, 224, 224), 
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