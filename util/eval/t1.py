import os, sys
from runner import *
import numpy as np
import signal

def sigint_handle(system:System):
    if system.mps_server != None:
        system.mps_server.kill()
    subprocess.run(['sudo', '/opt/mps-control/quit-mps-daemon.sh'])
    sys.exit(0)


global_seed = np.random.randint(2<<31)

infer_model_comm_config = {
    'path' : 'resnet152-b4',
    'device' : 'cuda',
    'batch-size' : '4',
    'num-worker' : '1'
}

def SimpleDemo(system:System, subdir_tag:str, infer_only:bool, hybrid:bool, local_seed = None):
    system.infer_model_config = {'resnet152' : infer_model_comm_config}
    workload = Workload(duration=60, concurrency=16, client_log="client-log", workload_log="workload-log",
        infer=True, infer_model=["resnet152-dp"], 
        change_time_point={"resnet152" : (tp:=list(range(20, 60, 5)))},
        dynamic_poisson={"resnet152" : [0] + [10 if i % 2 == 0 else 0 for i in range(len(tp))]},
        train_model=["resnet"], num_epoch=15, batch_size=60,
        seed=local_seed if local_seed is not None else global_seed
    )

    if infer_only:
        workload.train = False
        workload.benchmark = False
        system.launch('simple-demo', f'infer-only-{subdir_tag}')
        workload.launch(system)
        system.stop()

    if hybrid:
        workload.train = True
        workload.benchmark = False
        system.launch('simple-demo', f'hybrid-{subdir_tag}')
        workload.launch(system)
        system.stop()

    system.infer_model_config = None


def Microbenchmark(system:System, subdir_tag:str, infer_only:bool, hybrid:bool, local_seed=None):
    system.infer_model_config = {'resnet152[20]' : infer_model_comm_config}
    workload = Workload(duration=90, concurrency=16, client_log="client-log", workload_log="workload-log",
        infer=True, infer_model=["resnet152[20]"], 
        change_time_point={"benchmark": list(range(30, 90, 5))},
        train_model=["resnet"], num_epoch=15, batch_size=60,
        benchmark_random_dynamic_poisson=100,
        seed=local_seed if local_seed is not None else global_seed
    )
    if infer_only:
        workload.train = False
        workload.benchmark = True
        system.launch('micro', f'infer-only-{subdir_tag}')
        workload.launch(system)
        system.stop()

    if hybrid:
        workload.train = True
        workload.benchmark = True
        system.launch('micro', f'hybrid-{subdir_tag}')
        workload.launch(system)
        system.stop()

    system.infer_model_config = None


system = System(
    mode=System.ServerMode.Normal, 
    use_sta=False, 
    cuda_memory_pool_gb="13.5", 
    profile_log="server-profile", 
    server_log="server-log",
    mps=True
)
signal.signal(signal.SIGINT, lambda sig, frame: sigint_handle(system))

# os.environ["STA_RAW_ALLOC_UNIFIED_MEMORY"] = "1"
# os.environ["TORCH_UNIFIED_MEMORY"] = "1"
# SimpleDemo(system, 'normal', False, True, local_seed=2445152754)
# Microbenchmark(system, 'normal-tmp',False, True, local_seed=2445152754)
# os.environ["STA_RAW_ALLOC_UNIFIED_MEMORY"] = "0"
# os.environ["TORCH_UNIFIED_MEMORY"] = "0"

# SimpleDemo(system, 'normal', True, True)
# Microbenchmark(system, 'normal',True, True)


system.mode = System.ServerMode.ColocateL2
infer_model_comm_config['num-worker'] = '0'
# SimpleDemo(system, 'strawman', True, True)
# Microbenchmark(system, 'strawman', True, True)

system.use_sta = True
SimpleDemo(system, 'op1-sta', False, True)
Microbenchmark(system, 'op1-sta', False, True)

system.mode = System.ServerMode.ColocateL1
SimpleDemo(system, 'op2-kill-adjust', False, True)
Microbenchmark(system, 'op2-kill-adjust', False, True)
