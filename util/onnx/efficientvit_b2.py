
import torch
import timm
import trt_converter
from trt_converter import TrtModelConfig

batch_size = 4
model_name = 'efficientvit_b2'
model = timm.create_model('efficientvit_b2.r224_in1k', pretrained=True)



torch.onnx.export(model, torch.rand(batch_size, 3, 224, 224), 
                  f"{model_name}.onnx",verbose=False,
                  input_names=["input0"], output_names=["output0"], 
                  export_params=True)


trt_converter.convert_to_tensorrt(model_name, batch_size, TrtModelConfig.Vison)
