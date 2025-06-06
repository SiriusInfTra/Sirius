import torch
from torchvision import models
import onnx
from tvm.driver import tvmc
import tempfile
import tarfile
import pathlib

batch_size = 1
model_store = "server/models"
tmp_dir = tempfile.gettempdir()

resnet152 = models.resnet152(weights=models.ResNet152_Weights.DEFAULT).eval()
torch.onnx.export(resnet152, torch.rand(batch_size, 3, 224, 224), 
                  f"{tmp_dir}/resnet152.onnx", verbose=True,
                  input_names=["input"], output_names=["output"], export_params=True)

# tvmc compile --target "cuda" \
#   --input-shape "input:[x,3,224,224]" --output resnet152.tar resnet152.onnx
tvmc_model = tvmc.load(f"{tmp_dir}/resnet152.onnx", 
                       shape_dict={'input':[batch_size, 3, 224, 224]})
tvmc.compile(tvmc_model=tvmc_model, target='cuda', package_path=f"{tmp_dir}/resnet152-tvm.tar")


model_store_path = f'{model_store}/resnet152-b{batch_size}' 
pathlib.Path(model_store_path).mkdir(parents=True, exist_ok=True)
tarfile.open(f"{tmp_dir}/resnet152-tvm.tar").extractall(model_store_path)