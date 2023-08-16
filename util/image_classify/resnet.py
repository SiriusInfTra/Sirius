import torch
from torchvision import models
import onnx

model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT).eval()
torch.onnx.export(model, torch.rand(16, 3, 224, 224), "resnet152.onnx", verbose=True,
                  input_names=["input"], output_names=["output"], export_params=True,)
                  # dynamic_axes={'input':[0], 'output':[0]})