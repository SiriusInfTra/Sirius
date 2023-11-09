import os, sys
from runner import *
import numpy as np
import signal

def sigint_handle(system:System):
    if system.mps_server != None:
        system.mps_server.kill()
    subprocess.run(['sudo', '/opt/mps-control/quit-mps-daemon.sh'])
    sys.exit(0)


# global_seed = np.random.randint(2<<31)
global_seed = 3876034152

infer_model_comm_config = {
    'path' : 'resnet152-b4',
    'device' : 'cuda',
    'batch-size' : '4',
    'num-worker' : '1'
}

def SimpleAzure(system:System, subdir_tag:str, infer_only:bool, hybrid:bool, local_seed = None):
    system.infer_model_config = {'resnet152' : infer_model_comm_config}
    workload = Workload(duration=60, concurrency=16, client_log="client-log", workload_log="workload-log",
        infer=True, infer_model=["resnet152-azure"], 
        trace_id=1, peak_request=10, period=10, period_duration=10,
        train_model=["resnet"], num_epoch=15, batch_size=60,
        seed=local_seed if local_seed is not None else global_seed
    )

    if infer_only:
        workload.train = False
        # workload.benchmark = False
        system.launch('simple-azure', f'infer-only-{subdir_tag}')
        workload.launch(system)
        system.stop()

    if hybrid:
        workload.train = True
        # workload.benchmark = False
        system.launch('simple-azure', f'hybrid-{subdir_tag}')
        workload.launch(system)
        system.stop()

    system.infer_model_config = None


def BenchmarkAzure(system:System, subdir_tag:str, infer_only:bool, hybrid:bool, local_seed=None):
    system.infer_model_config = {'resnet152[10]' : infer_model_comm_config}
    workload = Workload(duration=90, concurrency=64, client_log="client-log", workload_log="workload-log",
        infer=True, infer_model=["resnet152-azure[10]"], 
        trace_id=1, peak_request=100, period=144, period_duration=0.1,
        train_model=["resnet"], num_epoch=15, batch_size=60,
        seed=local_seed if local_seed is not None else global_seed
    )
    if infer_only:
        workload.train = False
        workload.benchmark = False
        system.launch('benchmark-azure', f'infer-only-{subdir_tag}')
        workload.launch(system)
        system.stop()

    if hybrid:
        workload.train = True
        workload.benchmark = False
        system.launch('benchmark-azure', f'hybrid-{subdir_tag}')
        workload.launch(system)
        system.stop()

    system.infer_model_config = None


system = System(
    mode=System.ServerMode.Normal, 
    use_sta=False, 
    cuda_memory_pool_gb="12.0", 
    profile_log="server-profile", 
    server_log="server-log",
    mps=False
)
signal.signal(signal.SIGINT, lambda sig, frame: sigint_handle(system))

# os.environ["STA_RAW_ALLOC_UNIFIED_MEMORY"] = "1"
# os.environ["TORCH_UNIFIED_MEMORY"] = "1"
# SimpleDemo(system, 'normal', False, True, local_seed=2445152754)
# Microbenchmark(system, 'normal-tmp',False, True, local_seed=2445152754)
# os.environ["STA_RAW_ALLOC_UNIFIED_MEMORY"] = "0"
# os.environ["TORCH_UNIFIED_MEMORY"] = "0"

# SimpleAzure(system, 'normal', True, True)
# BenchmarkAzure(system, 'normal',True, False)


system.mode = System.ServerMode.ColocateL2
infer_model_comm_config['num-worker'] = '0'
# SimpleAzure(system, 'strawman', True, True)
# BenchmarkAzure(system, 'strawman', True, True)

system.use_sta = True
# SimpleAzure(system, 'op1-sta', False, True)
BenchmarkAzure(system, 'op1-sta', True, False)

system.mode = System.ServerMode.ColocateL1
# SimpleAzure(system, 'op2-kill-adjust', False, True)
# BenchmarkAzure(system, 'op2-kill-adjust', False, True)
