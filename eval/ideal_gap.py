import os
from runner import *

import subprocess
import time

set_global_seed(42)

def run(system: System, workload: HyperWorkload, tag: str):
    system.launch("ideal-gap", tag, time_stamp=False, 
                  infer_model_config=System.InferModelConfig("resnet152[16]", "resnet152-b1", "1", "0"))
    workload.launch_workload(system)
    system.stop()
    time.sleep(1)

system = System(mode=System.ServerMode.ColocateL2, use_sta=False, mps=True)
workload = HyperWorkload(concurrency=2048, duration=30, delay_before_infer=30)
workload.set_train_workload(train_workload=TrainWorkload('resnet', 10, 60))
workload.set_infer_workloads(AzureInferWorkload(
    'workload_data/azurefunctions-dataset2019/function_durations_percentiles.anon.d01.csv',
    max_request_sec=100, interval_sec=1, period_num=30, func_num=100,
    model_list=InferModel.get_model_list("resnet152", 16)
))

run(system, workload, "strawman")

system = System(mode=System.ServerMode.Normal, use_sta=False, mps=True, colocate_skip_malloc=True)
run(system, workload, "no-memory-pressure")

system = System(mode=System.ServerMode.Normal, use_sta=False, mps=True, colocate_skip_malloc=True, colocate_skip_loading=True)
system = System(mode=System.ServerMode.Normal, use_sta=False, mps=True)
run(system, workload, "no-load-overhead")

os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = "60" # a empirical value
system = System(mode=System.ServerMode.Normal, use_sta=False, mps=True, train_mps_thread_percent=40)
run(system, workload, "no-sm-contention")

workload.set_train_workload(None)
run(system, workload, "no-colocation")


