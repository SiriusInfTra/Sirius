import os
os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = "60" # a empirical value

from runner import System, HyperWorkload
from runner import *

import subprocess
import time

set_global_seed(42)

class MemoryPressure():
    def __init__(self, mb: int) -> None:
        self.memory_pressure_mb = mb
        self.simulator = None

    def __enter__(self):
        self.simulator = subprocess.Popen(["./util/misc/simulate.out", "0", f"{self.memory_pressure_mb}"], env=os.environ.copy())
        time.sleep(1)

    def __exit__(self, type, value, trace):
        self.simulator.send_signal(subprocess.signal.SIGINT)
        self.simulator.wait()
        self.simulator = None


def run(system: System, hyper_workload:HyperWorkload, num_model:int, req_per_sec:int, tag:str):
    InferModel.reset_model_cnt()
    # hyper_workload.set_infer_workloads(PoissonInferWorkload(
    #     poisson_params=zip(InferModel.get_model_list("resnet152", num_model), 
    #                        num_model * [PoissonParam(0, req_per_sec)]),
    #     duration=20,
    # ))
    hyper_workload.set_train_workload(train_workload=TrainWorkload('resnet', 10, 60))
    # hyper_workload.set_train_workload(None)
    infer_model_config = System.InferModelConfig(f"resnet152[{num_model}]", "resnet152", "1")
    system.launch("um-mps-issue", f"{tag}-{num_model}", time_stamp=True, infer_model_config=infer_model_config)
    hyper_workload.launch_busy_loop(system, InferModel.get_model_list("resnet152", num_model))
    system.stop()
    time.sleep(1)


system = System(mode=System.ServerMode.Normal, use_sta=False, mps=True, train_mps_thread_percent=40)
workload = HyperWorkload(concurrency=1, duration=30, delay_before_infer=30)

run(system, workload, 1, 4, "ideal")
run(system, workload, 2, 4, "ideal")
run(system, workload, 4, 4, "ideal")
run(system, workload, 8, 4, "ideal")
run(system, workload, 16, 4, "ideal")

os.environ["TORCH_UNIFIED_MEMORY"] = "1"
os.environ["STA_RAW_ALLOC_UNIFIED_MEMORY"] = "1"
with MemoryPressure(4975):
    run(system, workload, 1, 4, "mempres")
    run(system, workload, 2, 4, "mempres")
    run(system, workload, 4, 4, "mempres")
    run(system, workload, 8, 4, "mempres")
    run(system, workload, 16, 4, "mempres")