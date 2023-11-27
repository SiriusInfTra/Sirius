
from __future__ import annotations
import datetime
import time
import numpy as np
import os
from os import PathLike
import pathlib
import subprocess
from typing import List, Dict, Optional
from types import NoneType

from workload import InferWorkloadBase, TrainWorkload, InferTraceDumper


class System:
    class ServerMode:
        Normal = "normal"
        ColocateL1 = "colocate-l1"
        ColocateL2 = "colocate-l2"

    def __init__(self, mode:str, use_sta:bool, cuda_memory_pool_gb:str,
                 profile_log:str = "profile-log", 
                 server_log:str = "server-log", 
                 port:str = "18080",
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

    def launch(self, name: str, subdir: Optional[str] = None, trace_cfg: Optional[os.PathLike[str]] = None):
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
            self.mps_server = subprocess.Popen(
                ['sudo', '/opt/mps-control/launch-mps-daemon-private.sh',
                 '--device', os.environ['CUDA_VISIBLE_DEVICES'], '--mps-pipe', os.environ['CUDA_MPS_PIPE_DIRECTORY']],
                stderr=subprocess.PIPE, stdout=subprocess.PIPE, env=os.environ.copy())
        else:
            cmd += ["--mps", "0"]
            self.mps_server = None

        self.cmd_trace.append(" ".join(cmd))
        print(" ".join(cmd))

        with open(server_log, "w") as log_file:
            self.server = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=os.environ.copy())

        while True:
            with open(server_log, "r") as log_file:
                if self.server.poll() is not None:
                    print(log_file.read())
                    self.quit_mps()
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
            self.quit_mps()
            self.mps_server.wait()
            self.mps_server = None
        self.cmd_trace.append(" ".join([
            'sudo', '/opt/mps-control/quit-mps-daemon-private.sh',
            '--mps-pipe', os.environ['CUDA_MPS_PIPE_DIRECTORY']
        ]))
        with open(f'{self.log_dir}/cmd-trace', 'w') as f:
            f.write("\n\n".join(self.cmd_trace))
        self.log_dir = None

    def quit_mps(self):
        quit_mps = subprocess.run([
            'sudo', '/opt/mps-control/quit-mps-daemon-private.sh', 
            '--mps-pipe', os.environ['CUDA_MPS_PIPE_DIRECTORY']], 
            capture_output=True, env=os.environ.copy())


class HyperWorkload:
    def __init__(self, 
                 concurrency:int, 
                 duration: Optional[int | float] = None, 
                 workload_log:str = "workload-log", 
                 client_log:str = "client-log", 
                 trace_cfg:str = "trace-cfg",
                 seed: Optional[int] = None, 
                 delay_before_infer: float = 0) -> None:
        self.infer_workloads: List[InferWorkloadBase] = []
        self.train_workload: NoneType | TrainWorkload = None
        self.duration = duration
        self.workload_log = workload_log
        self.client_log = client_log
        self.trace_cfg = trace_cfg
        self.concurrency = concurrency
        self.seed = seed
        self.delay_before_infer = delay_before_infer
    
    def launch(self, server: System, trace_cfg: Optional[PathLike] = None):
        assert server.server is not None
        cmd = [
            "./build/workload_launcher",
            "-p", server.port,
            "-c", str(self.concurrency),
            "--delay-before-infer", str(self.delay_before_infer)
        ]
        if self.duration is not None:
            cmd += ["-d", str(self.duration)]
        
        if trace_cfg is not None:
            cmd += ["--infer-trace", str(trace_cfg)]
        elif len(self.infer_workloads) > 0:
            trace_cfg = pathlib.Path(server.log_dir) / self.trace_cfg
            InferTraceDumper(self.infer_workloads, trace_cfg).dump()
            cmd += ["--infer-trace", str(trace_cfg)]
        else:
            cmd += ["--no-infer"]

        if self.train_workload is not None:
            cmd += ["--train"]
            for key, value in self.train_workload._asdict().items():
                cmd += ['--' + key.replace('_', '-'), str(value)]
        else:
            cmd += ["--no-train"]

        if self.seed is not None:
            cmd += ["--seed", str(self.seed)]

        workload_log = pathlib.Path(server.log_dir) / self.workload_log
        cmd += ['--log', str(workload_log)]
        cmd += ['-v', '1']

        server.cmd_trace.append(" ".join(cmd))
        print(" ".join(cmd))

        client_log = pathlib.Path(server.log_dir) / self.client_log
        with open(client_log, "w") as log_file:
            completed = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=os.environ.copy())
            if completed.returncode != 0:
                raise Exception(f"Workload exited with code {completed.returncode}")


