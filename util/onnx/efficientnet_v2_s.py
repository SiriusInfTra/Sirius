import torch
from torchvision import models

import trt_converter
from trt_converter import TrtModelConfig

batch_size = 4
model_name = 'efficientnet_v2_s'
efficientnet_v2_s = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT).eval()


torch.onnx.export(efficientnet_v2_s, torch.rand(batch_size, 3, 224, 224), 
                  f"{model_name}.onnx",verbose=False,
                  input_names=["input0"], output_names=["output0"], 
                  export_params=True,
                  dynamic_axes={"input0": {0: "batch_size"}, "output0": {0: "batch_size"}})

trt_converter.convert_to_tensorrt(model_name, batch_size, TrtModelConfig.Vison)