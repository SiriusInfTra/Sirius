import pathlib, shutil
import argparse
import subprocess
import datetime
import sys
import time
from typing import Optional, List, Dict
import tempfile
import itertools
import os
import pynvml


GPU_UUIDs = []
pynvml.nvmlInit()
for i in range(pynvml.nvmlDeviceGetCount()):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    GPU_UUIDs.append(pynvml.nvmlDeviceGetUUID(handle).decode())
print(GPU_UUIDs)

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
    CUDA_VISIBLE_DEVICES_UUID = []
    for id in CUDA_VISIBLE_DEVICES.split(","):
        try:
            CUDA_VISIBLE_DEVICES_UUID.append(GPU_UUIDs[int(id.strip())])
        except:
            CUDA_VISIBLE_DEVICES_UUID.append(id.strip())
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(CUDA_VISIBLE_DEVICES_UUID)

os.environ['GLOG_logtostderr'] = "1"

class System:
    class ServerMode:
        Normal = "normal"
        ColocateL1 = "colocate-l1"
        ColocateL2 = "colocate-l2"

    def __init__(self, mode:str, use_sta:bool, cuda_memory_pool_gb:str,
                 profile_log:str, server_log:str, port:str= "8080",
                 infer_model_config:Dict[str, Dict[str, str]]=None,
                 mps=True) -> None:
        self.mode = mode
        self.port = port
        self.use_sta = use_sta
        self.cuda_memory_pool_gb = cuda_memory_pool_gb
        self.profile_log = profile_log
        self.server_log = server_log
        self.port = port
        self.server:Optional[subprocess.Popen]= None
        self.log_dir:Optional[str] = None
        self.cmd_trace = []
        self.infer_model_config = infer_model_config
        self.infer_model_config_path = None
        self.mps = mps
        self.mps_server = None
        self.time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    
    def next_time_stamp(self):
        self.time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")

    def launch(self, name, subdir = None):
        if subdir is None:
            self.log_dir = pathlib.Path("log") / f'{name}-{self.time_stamp}'
        else:
            self.log_dir = pathlib.Path("log") / f'{name}-{self.time_stamp}' / subdir
        pathlib.Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        server_log = f"{self.log_dir}/{self.server_log}.log"
        profile_log = f"{self.log_dir}/{self.profile_log}.log"

        self.cmd_trace = []
        cmd = [
            "./build/colserve", 
            "-p", self.port, 
            "--mode", self.mode, 
            "--use-sta", "1" if self.use_sta else "0", 
            "--cuda-memory-pool-gb", self.cuda_memory_pool_gb, 
            "--profile-log", profile_log
        ]
        if self.infer_model_config is not None:
            self.infer_model_config_path = f'{(self.log_dir / "infer-model-config").absolute()}'
            with open(self.infer_model_config_path, "w") as f:
                for m, c in self.infer_model_config.items():
                    print(m, file=f)
                    for k, v in c.items():
                        print(f"  {k} {v}", file=f)
                    print()
            cmd += ["--infer-model-config", self.infer_model_config_path]

        # first launch mps
        if self.mps:
            self.cmd_trace.append(" ".join([
                "sudo", "/opt/mps-control/launch-mps-daemon-private.sh"
                "--device", os.environ['CUDA_VISIBLE_DEVICES'], "--mps-pipe", os.environ['CUDA_MPS_PIPE_DIRECTORY']
            ]))
            self.mps_server = self.server = subprocess.Popen(
                ['sudo', '/opt/mps-control/launch-mps-daemon-private.sh',
                 '--device', os.environ['CUDA_VISIBLE_DEVICES'], '--mps-pipe', os.environ['CUDA_MPS_PIPE_DIRECTORY']],
                stderr=subprocess.PIPE, stdout=subprocess.PIPE, env=os.environ.copy())
        else:
            cmd += ["--mps", "0"]
            self.mps_server = None

        self.cmd_trace.append(" ".join(cmd))
        print(" ".join(cmd))

        time.sleep(1)
        with open(server_log, "w") as log_file:
            self.server = subprocess.Popen(cmd, 
                stdout=log_file, stderr=subprocess.STDOUT, env=os.environ.copy())

        while True:
            with open(server_log, "r") as log_file:
                if self.server.poll() is not None:
                    print(log_file.read())
                    raise RuntimeError("Server exited")
                if "GRPCServer start" not in log_file.read():
                    time.sleep(0.5)
                else:
                    break
    
    def stop(self):
        if self.server is not None:
            self.server.send_signal(subprocess.signal.SIGINT)
            self.server = None
        self.infer_model_config_path = None
        if self.mps_server is not None:
            quit_mps = subprocess.run([
                'sudo', '/opt/mps-control/quit-mps-daemon-private.sh', 
                '--mps-pipe', os.environ['CUDA_MPS_PIPE_DIRECTORY']], 
                capture_output=True, env=os.environ.copy())
            self.mps_server.wait()
            self.mps_server = None
        self.cmd_trace.append(" ".join([
            'sudo', '/opt/mps-control/quit-mps-daemon-private.sh',
            '--mps-pipe', os.environ['CUDA_MPS_PIPE_DIRECTORY']
        ]))
        with open(f'{self.log_dir}/cmd-trace', 'w') as f:
            f.write("\n\n".join(self.cmd_trace))
        self.log_dir = None


class Workload:
    def __init__(self, duration:int, concurrency:int, 
                 client_log:str, workload_log:str,
                 infer:bool=False, infer_model:Optional[List[str]]=None, 
                 poisson:Optional[Dict[str, int]]=None,
                 change_time_point:Optional[Dict[str, List[int]]]=None,
                 dynamic_concurrency:Optional[Dict[str, List[int]]]=None,
                 dynamic_poisson:Optional[Dict[str, List[int]]]=None,
                 benchmark:bool=False, 
                 benchmark_random_dynamic_poisson:Optional[int]=None, 
                 seed:Optional[int]=None,
                 train:bool=False, train_model:Optional[List[str]]=None, 
                 num_epoch:Optional[int]=None, batch_size:Optional[int]=None):
        self.duration = duration
        self.concurrency = concurrency
        self.infer = infer
        self.infer_model = infer_model
        self.poisson = poisson
        self.change_time_point = change_time_point
        self.dynamic_concurrency = dynamic_concurrency
        self.dynamic_poisson = dynamic_poisson
        self.benchmark = benchmark
        self.random_dynamic_poisson = benchmark_random_dynamic_poisson
        self.seed = seed
        self.train = train
        self.train_model = train_model
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.client_log = client_log
        self.workload_log = workload_log

    def launch(self, server:System):
        assert server.server is not None
        cmd = [
            "./build/hybrid_workload",
            "-p", server.port,
            "-d", str(self.duration),
            "-c", str(self.concurrency),
        ]
        if self.infer:
            cmd += ["--infer"]
            if self.infer_model is not None:
                cmd += ["--infer-model"] + self.infer_model
            if self.poisson is not None:
                cmd += ["--poisson"] + list(itertools.chain(*self.poisson.items()))
            if self.change_time_point is not None:
                cmd += ["--change-time-point"]
                for i, (m, tps) in enumerate(item_list:=self.change_time_point.items()):
                    cmd += [m] + [str(tp) for tp in tps]
                    if i + 1 != len(item_list):
                        cmd += ["%%"]
            if self.dynamic_concurrency is not None:
                cmd += ["--dynamic-concurrency"]
                for i, (m, cs) in enumerate(item_list:=self.dynamic_concurrency.items()):
                    cmd += [m] + [str(c) for c in cs]
                    if i + 1 != len(item_list):
                        cmd += ["%%"]
            if self.dynamic_poisson is not None:
                cmd += ["--dynamic-poisson"]
                for i, (m, ps) in enumerate(item_list:=self.dynamic_poisson.items()):
                    cmd += [m] + [str(p) for p in ps]
                    if i + 1 != len(item_list):
                        cmd += ["%%"]
        else:
            cmd += ["--no-infer"]

        if self.train:
            cmd += ["--train"]
            if self.train_model is not None:
                cmd += ["--train-model"] + self.train_model
                assert self.num_epoch is not None
                assert self.batch_size is not None
                cmd += ["--num-epoch", str(self.num_epoch)]
                cmd += ["--batch-size", str(self.batch_size)]
        else:
            cmd += ["--no-train"]

        if self.benchmark:
            cmd += ["--benchmark"]
            assert self.random_dynamic_poisson is not None
            cmd += ["--random-dynamic-poisson", str(self.random_dynamic_poisson)]

        if self.seed is not None:
            cmd += ["--seed", str(self.seed)]

        workload_log = pathlib.Path(server.log_dir) / self.workload_log
        cmd += ['--log', f'{workload_log}']
        cmd += ['-v', '1']

        server.cmd_trace.append(" ".join(cmd))
        print(" ".join(cmd))

        client_log = pathlib.Path(server.log_dir) / self.client_log
        with open(client_log, "w") as log_file:
            completed = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=os.environ.copy())
            if completed.returncode != 0:
                raise Exception(f"Workload exited with code {completed.returncode}")

        
