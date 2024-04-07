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


batch_size = 1
model_store = "server/models"
tmp_dir = tempfile.gettempdir()

# vit_s_16 = timm.create_model('vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=True).eval()

# vit_s_16 = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT).eval()
# traced_model = torch.jit.script(vit_b_16, torch.rand(batch_size, 3, 224, 224))
vit_s_16 = ViTForImageClassification.from_pretrained('WinKawaks/vit-small-patch16-224')
torch.onnx.export(vit_s_16, torch.rand(batch_size, 3, 224, 224), 
                  f"{tmp_dir}/vit_s_16.onnx",verbose=False,
                  input_names=["input"], output_names=["output"], 
                  export_params=True)

# tvmc compile --target "cuda" \
#   --input-shape "input:[x,3,224,224]" --output resnet152.tar resnet152.onnx

tvmc_model = tvmc.load(f"{tmp_dir}/vit_s_16.onnx", 
                       shape_dict={'input':[batch_size, 3, 224, 224]})

# tune_records = f'{tmp_dir}/vit_s_16-tune.json'
# tvmc.tune(tvmc_model=tvmc_model, target='cuda', 
#           tuning_records=tune_records, 
#           enable_autoscheduler=True)

tune_records = "util/prepare_model_store/vit_s_16-tune.json"
tvmc.compile(tvmc_model=tvmc_model, target='cuda', package_path=f"{tmp_dir}/vit_s_16-tvm.tar",
             tuning_records=tune_records)


model_store_path = f'{model_store}/vit_s_16-b{batch_size}' 
pathlib.Path(model_store_path).mkdir(parents=True, exist_ok=True)
tarfile.open(f"{tmp_dir}/vit_s_16-tvm.tar").extractall(model_store_path)