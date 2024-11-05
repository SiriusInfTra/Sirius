import os
from typing import cast
import torch
from torchvision import models
from tvm.driver import tvmc
import tempfile
import tarfile
import pathlib
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--tune", action="store_true")
parser.add_argument("--model-store", type=str, default="server/models")
args = parser.parse_args()

batch_size = cast(int, args.batch_size)
model_store = cast(str, args.model_store)
tune = cast(bool, args.tune)

tmp_dir = tempfile.gettempdir()
onnx_path = f"{tmp_dir}/modeltmp-densenet161-b{batch_size}.onnx"
tvm_path = f"{tmp_dir}/modeltmp-densenet161-b{batch_size}.tar"
model_store_path = f'{model_store}/densenet161-b{batch_size}' 
tune_records = os.path.join(os.path.dirname(__file__), 'tune_records',
                            f'densenet161-b{batch_size}-tune.json')

print('Preparing model...')
if not os.path.exists(onnx_path):
    densenet161 = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT).eval()
    torch.onnx.export(densenet161, torch.rand(batch_size, 3, 224, 224), 
                    onnx_path, verbose=True,
                    input_names=["input"], output_names=["output"], export_params=True)

print('Tuning model...')
tvmc_model = tvmc.load(onnx_path, shape_dict={'input':[batch_size, 3, 224, 224]})

if tune:
    tvmc.tune(tvmc_model=tvmc_model, target='cuda', 
            tuning_records=tune_records, 
            enable_autoscheduler=False)
else:
    tune_records = None

print('Compiling model...')
tvmc.compile(tvmc_model=tvmc_model, target='cuda', package_path=tvm_path,
             tuning_records=tune_records)
print('Extracting model...')
pathlib.Path(model_store_path).mkdir(parents=True, exist_ok=True)
tarfile.open(tvm_path).extractall(model_store_path)
print('Done!')