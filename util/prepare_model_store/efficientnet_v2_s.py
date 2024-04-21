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
model_store = "server/models"
tmp_dir = tempfile.gettempdir()


model_name = 'efficientnet_v2_s'
efficientnet_v2_s = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT).eval()


torch.onnx.export(efficientnet_v2_s, torch.rand(batch_size, 3, 224, 224), 
                  f"{tmp_dir}/{model_name}.onnx",verbose=False,
                  input_names=["input"], output_names=["output"], 
                  export_params=True)

tvmc_model = tvmc.load(f"{tmp_dir}/{model_name}.onnx", 
                       shape_dict={'input':[batch_size, 3, 224, 224]})

# tune_records = f'./{model_name}-tune-{platform}.json'
# tvmc.tune(tvmc_model=tvmc_model, target='cuda', 
#           tuning_records=tune_records, 
#           enable_autoscheduler=False,
#           port=9999)

tune_records = f"util/prepare_model_store/{model_name}-tune-{platform}.json"

tvmc.compile(tvmc_model=tvmc_model, target='cuda', package_path=f"{tmp_dir}/{model_name}-tvm.tar",
             tuning_records=tune_records)


model_store_path = f'{model_store}/{model_name}-b{batch_size}' 
pathlib.Path(model_store_path).mkdir(parents=True, exist_ok=True)
tarfile.open(f"{tmp_dir}/{model_name}-tvm.tar").extractall(model_store_path)