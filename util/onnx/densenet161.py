import torch
from torchvision import models

import trt_converter
from trt_converter import TrtModelConfig


model_name = "densenet161"
batch_size = 4
densenet161 = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT).eval()
torch.onnx.export(
    densenet161, torch.rand(batch_size, 3, 224, 224), 
    f"{model_name}.onnx", verbose=True,
    input_names=["input0"], output_names=["output0"], export_params=True,
    dynamic_axes={"input0": {0: "batch_size"}, "output0": {0: "batch_size"}})
trt_converter.convert_to_tensorrt(model_name, batch_size, TrtModelConfig.Vison)