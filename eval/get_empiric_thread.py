import os

from runner import System, HyperWorkload
from runner import *
from pathlib import Path
import time
import subprocess

set_global_seed(42)


def run(system: System, hyper_workload:HyperWorkload, num_model:int, req_per_sec:float, tag:str):
    InferModel.reset_model_cnt()
    infer_model_config = System.InferModelConfig(f"resnet152[{num_model}]", "resnet152", "1")
    system.launch("taskswitch-issue", f"{tag}-{req_per_sec}", time_stamp=True, infer_model_config=infer_model_config)
    hyper_workload.set_infer_workloads(PoissonInferWorkload(
        poisson_params=list(zip(InferModel.get_model_list("resnet152", num_model), [PoissonParam(0, req_per_sec / num_model)] * num_model)),
        duration=60, 
    ))
    hyper_workload.launch_workload(system)
    system.stop()
    time.sleep(1)
    
def run_with_dcgmi(system: System, hyper_workload:HyperWorkload, num_model:int, req_per_sec:int, tag:str):
    dcgmi_output_dir = Path("dcgmi_output")
    dcgmi_output_dir.mkdir(exist_ok=True)
    with open(dcgmi_output_dir / f"get-empiric-thread-{num_model}-{req_per_sec}-{tag}.txt", "w") as f:

        p = subprocess.Popen("dcgmi dmon -i 2 -e 1002".split(), stderr=sys.stderr, stdout=f)
        run(system, hyper_workload, num_model, req_per_sec, tag)
        p.terminate()
        p.communicate()
    
def main():
    system = System(mode=System.ServerMode.Normal, use_sta=False, mps=False)
    workload = HyperWorkload(concurrency=1, duration=10, wait_train_setup_sec=0)
    run_with_dcgmi(system, workload, 1, 0.1, "ideal")
    run_with_dcgmi(system, workload, 1, 0.2, "ideal")
    run_with_dcgmi(system, workload, 1, 0.5, "ideal")
    run_with_dcgmi(system, workload, 1, 1, "ideal")
    run_with_dcgmi(system, workload, 1, 2, "ideal")
    run_with_dcgmi(system, workload, 1, 4, "ideal")
    run_with_dcgmi(system, workload, 1, 8, "ideal")
    run_with_dcgmi(system, workload, 1, 16, "ideal")
    run_with_dcgmi(system, workload, 1, 32, "ideal")


if __name__ == "__main__":
    main()