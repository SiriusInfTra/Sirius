import sys
sys.path.append("./eval/")
import time
from runner import *
import argparse

set_global_seed(42)
use_time_stamp = True


infer_model = InferModel.ResNet152

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='resnet152')
args = parser.parse_args()

if 'resnet152' in args.model:
    infer_model = InferModel.ResNet152
elif 'efficientvit' in args.model:
    infer_model = InferModel.EfficientViT_b2
elif 'efficientnet' in args.model:
    infer_model = InferModel.EfficientNetV2_s
elif 'distilbert' in args.model:
    infer_model = InferModel.DistilBertBase
elif 'distilgpt2' in args.model:
    infer_model = InferModel.DistilGPT2
elif 'densenet161' in args.model:
    infer_model = InferModel.DenseNet161
else:
    raise ValueError(f"Invalid model: {args.model}")


system = System(mode=System.ServerMode.ColocateL1, 
                use_sta=True, use_xsched=False, mps=False,
                pipeline_load=False, port="18480",
                cuda_memory_pool_gb="13.5", 
                has_warmup=True)
workload = HyperWorkload(concurrency=1, duration=5, 
                         warmup=5, 
                         wait_warmup_done_sec=5,
                         wait_stable_before_start_profiling_sec=0)
InferModel.reset_model_cnt()
# infer_model_config = InferModel.get_model_list("resnet152", 1)
client_model_list, server_model_config = InferModel.get_multi_model([infer_model], 1, 1)

system.launch("infer-loading", f"{infer_model}", time_stamp=use_time_stamp, 
              infer_model_config=server_model_config)
workload.launch_busy_loop(system, client_model_list)
system.stop()

time.sleep(1)