import torch
from torchvision import models
import tempfile
import trt_converter
from trt_converter import TrtModelConfig

batch_size = 4
model_store = "server/models"
model_name = "resnet152"
tmp_dir = tempfile.gettempdir()

resnet152 = models.resnet152(weights=models.ResNet152_Weights.DEFAULT).eval()
torch.onnx.export(resnet152, torch.rand(batch_size, 3, 224, 224), 
                  f"{model_name}.onnx", verbose=True,
                  input_names=["input0"], output_names=["output0"], export_params=True,
                  dynamic_axes={"input0": {0: "batch_size"}, "output0": {0: "batch_size"}})

trt_converter.convert_to_tensorrt(model_name, batch_size,TrtModelConfig.Vison)