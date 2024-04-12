import os

from runner import System, HyperWorkload
from runner import *
import time

set_global_seed(42)


def run(system: System, hyper_workload:HyperWorkload, num_model:int, req_per_sec:int, tag:str, time_stamp: str):
    InferModel.reset_model_cnt()
    # hyper_workload.set_infer_workloads(PoissonInferWorkload(
    #     poisson_params=zip(InferModel.get_model_list("resnet152", num_model), 
    #                        num_model * [PoissonParam(0, req_per_sec)]),
    #     duration=20,
    # ))
    hyper_workload.set_train_workload(train_workload=TrainWorkload('resnet', 15, 32))
    # hyper_workload.set_train_workload(None)
    infer_model_config = System.InferModelConfig(f"resnet152[{num_model}]", "resnet152", "1")
    req_per_sec_str = "%.2d" % req_per_sec if isinstance(req_per_sec, int) else "%.2f" % req_per_sec
    system.launch(f"taskswitch-issue-{time_stamp}", f"{tag}-{req_per_sec_str}", time_stamp=False, infer_model_config=infer_model_config)
    hyper_workload.set_infer_workloads(PoissonInferWorkload(
        poisson_params=list(zip(InferModel.get_model_list("resnet152", num_model), [PoissonParam(0, req_per_sec / num_model)] * num_model)),
        duration=60, 
    ))
    hyper_workload.launch_workload(system)
    system.stop()
    time.sleep(1)


def main():
    # with SetEnvs({'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE': '60'}):

    parmas = [
        (1, 0.1, 98),
        (1, 0.2, 98),
        (1, 0.5, 98),
        (1, 1,   97),
        (1, 2,   96),
        (1, 4,   92),
        (1, 8,   90),
        (1, 16,  84),
        (1, 32,  76),
    ]
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    for num_model, req_per_sec, train_thread_percent in parmas:
        system = System(mode=System.ServerMode.Normal, use_sta=False, mps=True, train_mps_thread_percent=train_thread_percent)
        workload = HyperWorkload(concurrency=1, duration=60, wait_train_setup_sec=10)
        run(system, workload, num_model, req_per_sec, "ideal", timestamp)

    system = System(mode=System.ServerMode.TaskSwitchL1, use_sta=False, mps=False)
    workload = HyperWorkload(concurrency=32, duration=60, wait_train_setup_sec=30)
    run(system, workload, 1, 0.1, "taskswitch", timestamp)
    run(system, workload, 1, 0.2, "taskswitch", timestamp)
    run(system, workload, 1, 0.5, "taskswitch", timestamp)
    run(system, workload, 1, 1, "taskswitch", timestamp)
    run(system, workload, 1, 1, "taskswitch", timestamp)
    run(system, workload, 1, 2, "taskswitch", timestamp)
    run(system, workload, 1, 4, "taskswitch", timestamp)
    run(system, workload, 1, 8, "taskswitch", timestamp)
    run(system, workload, 1, 16, "taskswitch", timestamp)
    run(system, workload, 1, 32, "taskswitch", timestamp)


if __name__ == "__main__":
    main()