import os
from runner import *

import subprocess
import time

set_global_seed(42)

def run(system: System, workload: HyperWorkload, num_worker:int ,tag: str):
    system.launch("ideal-gap", tag, time_stamp=True, 
                  infer_model_config=System.InferModelConfig("resnet152[16]", "resnet152-b1", 1, num_worker))
    workload.launch_workload(system)
    system.stop()
    time.sleep(1)

system = System(mode=System.ServerMode.ColocateL2, use_sta=False, mps=True, keep_last_time_stamp=True, infer_blob_alloc=False)
hyper_workload = HyperWorkload(concurrency=2048, duration=30, wait_train_setup_sec=30)
hyper_workload.set_train_workload(train_workload=TrainWorkload('resnet', num_epoch=20, batch_size=56))
# workload.set_infer_workloads(AzureInferWorkload(
#     AzureInferWorkload.TRACE_D01,
#     max_request_sec=100, interval_sec=120 / 720, period_num=720, func_num=16 * 1000,
#     model_list=InferModel.get_model_list("resnet152", 16)
# ))
hyper_workload.set_infer_workloads(MicrobenchmarkInferWorkload(
    model_list=InferModel.get_model_list("resnet152", 16),
    max_request_sec=100, interval_sec=1, duration=60, 
))

run(system, hyper_workload, 0, "strawman")

system = System(mode=System.ServerMode.ColocateL2, use_sta=False, mps=True, 
                colocate_skip_malloc=True, 
                keep_last_time_stamp=True)
run(system, hyper_workload, 0, "no-memory-pressure")

system = System(mode=System.ServerMode.ColocateL2, use_sta=False, mps=True, 
                colocate_skip_malloc=True, colocate_skip_loading=True,
                keep_last_time_stamp=True)
run(system, hyper_workload, 0, "no-load-overhead")

os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = "60" # a empirical value
system = System(mode=System.ServerMode.ColocateL2, use_sta=False, mps=True, 
                colocate_skip_malloc=True, colocate_skip_loading=True, train_mps_thread_percent=40,
                keep_last_time_stamp=True)
run(system, hyper_workload, 0, "no-sm-contention")

os.environ.pop("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")
system = System(mode=System.ServerMode.Normal, use_sta=False, mps=True, train_mps_thread_percent=40,
                keep_last_time_stamp=True)
hyper_workload.set_train_workload(None)
run(system, hyper_workload, 1, "no-colocation")


