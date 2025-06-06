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

vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT).eval()
torch.onnx.export(vgg19, torch.rand(batch_size, 3, 224, 224), 
                  f"{tmp_dir}/vgg19.onnx", verbose=True,
                  input_names=["input"], output_names=["output"], export_params=True,
                  dynamic_axes={'input':[0], 'output':[0]})

# tvmc compile --target "cuda" \
#   --input-shape "input:[x,3,224,224]" --output resnet50.tar resnet50.onnx
tvmc_model = tvmc.load(f"{tmp_dir}/vgg19.onnx", 
                       shape_dict={'input':[batch_size, 3, 224, 224]})
tvmc.compile(tvmc_model=tvmc_model, target='cuda', package_path=f"{tmp_dir}/vgg19-tvm.tar")


model_store_path = f'{model_store}/vgg19-b{batch_size}' 
pathlib.Path(model_store_path).mkdir(parents=True, exist_ok=True)
tarfile.open(f"{tmp_dir}/vgg19-tvm.tar").extractall(model_store_path)