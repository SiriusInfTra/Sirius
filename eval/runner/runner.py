
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
from dataclasses import dataclass

from .workload import InferWorkloadBase, TrainWorkload, InferTraceDumper, InferModel, RandomInferWorkload
from .config import get_global_seed


class System:
    _last_time_stamp = None

    class ServerMode:
        Normal = "normal"
        ColocateL1 = "colocate-l1"
        ColocateL2 = "colocate-l2"

    @dataclass
    class InferModelConfig:
        model_name: str
        path: str
        batch_size: int
        num_worker: int = 1
        device: str = "cuda"
        max_worker: int = 1

        @classmethod
        def Empty(cls):
            return cls(None, None, None)

        def __repr__(self) -> str:
            if self.model_name is None:
                return ""
            return f'''{self.model_name}
  path {self.path}
  device {self.device}
  batch-size {self.batch_size}
  num-worker {self.num_worker}
  max-worker {self.max_worker}
'''

    def __init__(self, mode: str, use_sta: bool, 
                 cuda_memory_pool_gb: str=None,
                 profile_log: str = "profile-log", 
                 server_log: str = "server-log", 
                 port: str = "18080",
                 infer_model_config: List[InferModelConfig] | InferModelConfig = None,
                 mps: bool = True,
                 infer_blob_alloc: bool = False,
                 train_mps_thread_percent: Optional[int] = None,
                 colocate_skip_malloc: bool = False,
                 colocate_skip_loading: bool = False,
                 keep_last_time_stamp: bool = False) -> None:
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
        if infer_model_config is not None:
            if isinstance(infer_model_config, System.InferModelConfig):
                infer_model_config = [infer_model_config]
            self.infer_model_config = infer_model_config 
        else:
            self.infer_model_config = None
        self.infer_model_config_path = None
        self.mps = mps
        self.mps_server = None
        self.infer_blob_alloc = infer_blob_alloc
        self.train_mps_thread_percent = train_mps_thread_percent
        self.colocate_skip_malloc = colocate_skip_malloc
        self.colocate_skip_loading = colocate_skip_loading
        if System._last_time_stamp is None or not keep_last_time_stamp:
            self.time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
            System._last_time_stamp = self.time_stamp
        else:
            self.time_stamp = System._last_time_stamp

    def next_time_stamp(self):
        self.time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")

    def launch(self, name: str, subdir: Optional[str] = None, time_stamp:bool=True, 
               infer_model_config: List[InferModelConfig] | InferModelConfig = None):
        if subdir is None:
            if time_stamp:
                self.log_dir = pathlib.Path("log") / f'{name}-{self.time_stamp}'
            else:
                self.log_dir = pathlib.Path("log") / f'{name}'
        else:
            if time_stamp:
                self.log_dir = pathlib.Path("log") / f'{name}-{self.time_stamp}' / subdir
            else:
                self.log_dir = pathlib.Path("log") / f'{name}' / subdir
        pathlib.Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        server_log = f"{self.log_dir}/{self.server_log}.log"
        profile_log = f"{self.log_dir}/{self.profile_log}.log"

        self.cmd_trace = []
        cmd = [
            "./build/colserve", 
            "-p", self.port, 
            "--mode", self.mode, 
            "--use-sta", "1" if self.use_sta else "0", 
            "--profile-log", profile_log
        ]
        if self.cuda_memory_pool_gb is not None:
            cmd += ["--cuda-memory-pool-gb", self.cuda_memory_pool_gb]
        if infer_model_config is not None:
            if isinstance(infer_model_config, System.InferModelConfig):
                infer_model_config = [infer_model_config]
            self.infer_model_config = infer_model_config
        if self.infer_model_config is not None:
            self.infer_model_config_path = f'{(self.log_dir / "infer-model-config").absolute()}'
            with open(self.infer_model_config_path, "w") as f:
                for config in self.infer_model_config:
                    print(config, end="\n\n", file=f)
            cmd += ["--infer-model-config", self.infer_model_config_path]

        # first launch mps
        if self.mps:
            cmd += ["--mps", "1"]
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

        if self.infer_blob_alloc:
            cmd += ["--infer-blob-alloc"]

        if self.train_mps_thread_percent is not None:
            cmd += ["--train-mps-thread-percent", str(self.train_mps_thread_percent)]

        if self.colocate_skip_malloc:
            cmd += ["--colocate-skip-malloc"]
        if self.colocate_skip_loading:
            cmd += ["--colocate-skip-loading"]

        self.cmd_trace.append(" ".join(cmd))
        print("\n---------------------------\n")
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
                 delay_before_infer: float = 0,
                 warmup: int = 0,
                 delay_after_warmup: Optional[float] = None) -> None:
        self.enable_infer = True
        self.enable_train = True
        self.infer_workloads: List[InferWorkloadBase] = []
        self.infer_models: List[InferModel] = []
        self.train_workload: NoneType | TrainWorkload = None
        self.duration = duration
        self.workload_log = workload_log
        self.client_log = client_log
        self.trace_cfg = trace_cfg
        self.concurrency = concurrency
        if seed is not None:
            self.seed = seed
        else:
            self.seed = get_global_seed()
        self.delay_before_infer = delay_before_infer
        self.warmup = warmup
        self.delay_after_warmup = delay_after_warmup

    def set_infer_workloads(self, *infer_workloads: InferWorkloadBase):
        self.infer_workloads = list(infer_workloads)
    
    def set_train_workload(self, train_workload: TrainWorkload | NoneType):
        self.train_workload = train_workload

    def disable_infer(self):
        self.enable_infer = False

    def disable_train(self):
        self.enable_train = False

    def launch_workload(self, server: System, trace_cfg: Optional[PathLike] = None):
        infer_trace_args = []
        if trace_cfg is not None:
            infer_trace_args += ["--infer-trace", str(trace_cfg)]
        elif len(self.infer_workloads) > 0:
            for workload in self.infer_workloads:
                if isinstance(workload, RandomInferWorkload):
                    workload.reset_random_state()
            trace_cfg = pathlib.Path(server.log_dir) / self.trace_cfg
            InferTraceDumper(self.infer_workloads, trace_cfg).dump()
            infer_trace_args += ["--infer-trace", str(trace_cfg)]

        self._launch(server, "workload_launcher", infer_trace_args)

    def launch_busy_loop(self, server: System, infer_models: List[InferModel] = None):
        if infer_models is None:
            infer_models = self.infer_models
        infer_model_args = []
        if len(infer_models) > 0:
            infer_model_args += ["--infer-model"]
            for model in infer_models:
                infer_model_args += [model.model_name]

        self._launch(server, "busy_loop_launcher", infer_model_args)
        
    def _launch(self, server: System, launcher: str, custom_args: List[str] = []):
        assert server is not None
        cmd = [
            f"./build/{launcher}",
            "-p", server.port,
            "-c", str(self.concurrency),
            "--delay-before-infer", str(self.delay_before_infer)
        ]
        if self.duration is not None:
            cmd += ["-d", str(self.duration)]

        if self.enable_infer:
            cmd += ["--infer"]
        else:
            cmd += ["--no-infer"]

        if self.enable_train and self.train_workload is not None:
            cmd += ["--train"]
            for key, value in self.train_workload._asdict().items():
                cmd += ['--' + key.replace('_', '-'), str(value)]
        else:
            cmd += ["--no-train"]

        assert self.seed is not None
        cmd += ["--seed", str(self.seed)]

        cmd += ["--warmup", str(self.warmup)]
        if self.warmup > 0 and self.delay_after_warmup is not None:
            cmd += ["--delay-after-warmup", str(self.delay_after_warmup)]

        workload_log = pathlib.Path(server.log_dir) / self.workload_log
        cmd += ['--log', str(workload_log)]
        cmd += ['-v', '1']

        cmd += custom_args

        server.cmd_trace.append(" ".join(cmd))
        print(" ".join(cmd))

        try:
            client_log = pathlib.Path(server.log_dir) / self.client_log
            with open(client_log, "w") as log_file:
                completed = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=os.environ.copy())
                if completed.returncode != 0:
                    raise Exception(f"Workload exited with code {completed.returncode}")
        except Exception as e:
            print(f"Workload exited with exception")
            server.stop()
            raise e
