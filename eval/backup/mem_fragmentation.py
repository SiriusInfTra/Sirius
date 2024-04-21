import os

from runner import System, HyperWorkload
from runner import *
import time

set_global_seed(42)




def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")

    system = System(mode=System.ServerMode.ColocateL1, use_sta=True, mps=True, 
                    cuda_memory_pool_gb="12", ondemand_adjust=False)
    workload = HyperWorkload(concurrency=32, duration=60, wait_train_setup_sec=90)
    InferModel.reset_model_cnt()
    workload.set_train_workload(train_workload=TrainWorkload('resnet', 45, 64))
    num_model = 4
    req_per_sec = 32
    infer_model_config = System.InferModelConfig(f"resnet152[{num_model}]", "resnet152", "1", 0)
    req_per_sec_str = "%.2d" % req_per_sec if isinstance(req_per_sec, int) else "%.2f" % req_per_sec
    system.launch(f"mem-fragmentation-{timestamp}", f"{req_per_sec_str}", time_stamp=False, 
                  infer_model_config=infer_model_config)
    workload.set_infer_workloads(AzureInferWorkload(
        AzureInferWorkload.TRACE_D01,
        max_request_sec=30, interval_sec=120 / 720, period_num=720, func_num=16 * 1000, 
        model_list=InferModel.get_model_list("resnet152", num_model),
    ))
    workload.launch_workload(system)
    system.stop()
    time.sleep(1)


if __name__ == "__main__":
    main()