from runner import *

set_global_seed(42)

use_time_stamp = False

@dataclass
class EvalConfig:
    server_port = "18590"

eval_config = EvalConfig()


def train_only():
    workload = HyperWorkload(duration=30, concurrency=2048,
                             wait_train_setup_sec=5,)
    workload.set_train_workload(train_workload=TrainWorkload('resnet', 5, 48))
    return workload


def run(system: System, workload: HyperWorkload):
    system.launch("clean-release", None, time_stamp=use_time_stamp)
    workload.launch_workload(system)
    system.stop()
    time.sleep(3)
    system.draw_memory_usage()


hyper_workload = train_only()
system = System(mode=System.ServerMode.ColocateL1, use_sta=True, mps=False, use_xsched=True, 
                port=eval_config.server_port, cuda_memory_pool_gb="13.5", dummy_adjust=True)
run(system, hyper_workload)


