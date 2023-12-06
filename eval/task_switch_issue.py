import os

from runner import System, HyperWorkload
from runner import *
import time

set_global_seed(42)


def run(system: System, hyper_workload:HyperWorkload, num_model:int, req_per_sec:int, tag:str):
    InferModel.reset_model_cnt()
    # hyper_workload.set_infer_workloads(PoissonInferWorkload(
    #     poisson_params=zip(InferModel.get_model_list("resnet152", num_model), 
    #                        num_model * [PoissonParam(0, req_per_sec)]),
    #     duration=20,
    # ))
    hyper_workload.set_train_workload(train_workload=TrainWorkload('resnet', 15, 32))
    # hyper_workload.set_train_workload(None)
    infer_model_config = System.InferModelConfig(f"resnet152[{num_model}]", "resnet152", "1")
    system.launch("taskswitch-issue", f"{tag}-{req_per_sec:02d}", time_stamp=True, infer_model_config=infer_model_config)
    hyper_workload.set_infer_workloads(PoissonInferWorkload(
        poisson_params=list(zip(InferModel.get_model_list("resnet152", num_model), [PoissonParam(0, req_per_sec / num_model)] * num_model)),
        duration=60, 
    ))
    hyper_workload.launch_workload(system)
    system.stop()
    time.sleep(1)


# system = System(mode=System.ServerMode.Normal, use_sta=False, mps=True, train_mps_thread_percent=40)
# workload = HyperWorkload(concurrency=1, duration=60, delay_before_infer=10)

# run(system, workload, 1, 1, "ideal")
# run(system, workload, 1, 2, "ideal")
# run(system, workload, 1, 4, "ideal")
# run(system, workload, 1, 8, "ideal")
# run(system, workload, 1, 16, "ideal")
# run(system, workload, 1, 32, "ideal")

system = System(mode=System.ServerMode.TaskSwitchL1, use_sta=False, mps=False)
workload = HyperWorkload(concurrency=32, duration=60, delay_before_infer=30)

run(system, workload, 1, 1, "taskswitch")
run(system, workload, 1, 2, "taskswitch")
run(system, workload, 1, 4, "taskswitch")
run(system, workload, 1, 8, "taskswitch")
run(system, workload, 1, 16, "taskswitch")
run(system, workload, 1, 32, "taskswitch")
