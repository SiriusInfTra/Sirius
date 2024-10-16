
from __future__ import annotations
import contextlib
import datetime
import signal
import time
import numpy as np
import os, io
from os import PathLike
import pathlib
import subprocess
from typing import List, Dict, Optional
from types import NoneType
from dataclasses import dataclass
from .triton_cp import cp_model
import re

from .hyper_workload import (
    InferWorkloadBase, TrainWorkload, 
    InferTraceDumper, InferModel, RandomInferWorkload
)
from .config import (
    get_global_seed, get_binary_dir,
    get_host_name, is_meepo5
)

def get_num_gpu():
    cuda_device_env = os.environ['CUDA_VISIBLE_DEVICES']
    return len(cuda_device_env.split(','))


class CudaMpsCtrl:
    @classmethod
    def launch_cmd(cls, 
                   device: Optional[str] = None, 
                   mps_pipe: Optional[str] = None):
        if device is None:
                device = os.environ['CUDA_VISIBLE_DEVICES']
        if mps_pipe is None:
            mps_pipe = os.environ['CUDA_MPS_PIPE_DIRECTORY']

        if is_meepo5():
            return [
                "sudo", "/opt/mps-control/launch-mps-daemon-private.sh",
                "--device", device, "--mps-pipe", mps_pipe
            ]
        elif pathlib.Path('/.dockerenv').exists():
            return [
                "./scripts/docker_launch_mps.sh",
                "--device", device, "--mps-pipe", mps_pipe
            ]
        else:
            raise RuntimeError("Unknown MPS launch environment")

    @classmethod
    def quit_cmd(cls, mps_pipe: Optional[str] = None):
        if mps_pipe is None:
            mps_pipe = os.environ['CUDA_MPS_PIPE_DIRECTORY']
        
        if is_meepo5():
            return [
                'sudo', '/opt/mps-control/quit-mps-daemon-private.sh',
                '--mps-pipe', mps_pipe
            ]
        elif pathlib.Path('/.dockerenv').exists():
            return [
                "./scripts/docker_quit_mps.sh",
                "--mps-pipe", mps_pipe
            ]


class System:
    _last_time_stamp = None

    class ServerMode:
        Normal = "normal"
        ColocateL1 = "colocate-l1"
        ColocateL2 = "colocate-l2"
        TaskSwitchL1 = "task-switch-l1"
        TaskSwitchL2 = "task-switch-l2"
        TaskSwitchL3 = "task-switch-l3"
    
    class MemoryPoolPolicy:
        FirstFit = "first-fit"
        NextFit = "next-fit"
        BestFit = "best-fit"

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
                 use_sta_train: Optional[bool] = None,
                 cuda_memory_pool_gb: str = None,
                 memory_pool_policy: str = MemoryPoolPolicy.BestFit,
                 profile_log: str = "profile-log", 
                 server_log: str = "server-log", 
                 triton_log: str = "triton_log",
                 mps_log: str = "mps_log",
                 train_profile: str = "train-profile", 
                 port: str = "18080",
                 use_triton: bool = False,
                 infer_model_config: List[InferModelConfig] | InferModelConfig = None,
                 mps: bool = True,
                 skip_set_mps_thread_percent: bool = False,
                 use_xsched: bool = False,
                 dynamic_sm_partition: bool = False,
                 infer_blob_alloc: bool = False,
                 train_mps_thread_percent: Optional[int] = None,
                 colocate_skip_malloc: bool = False,
                 colocate_skip_loading: bool = False,
                 max_warm_cache_nbytes: int = 0 * 1024 * 1024 * 1024,
                 cold_cache_min_capability_nbytes: int = 0 * 1024 * 1024 * 1024,
                 cold_cache_max_capability_nbytes: int = 0 * 1024 * 1024 * 1024,
                 cold_cache_ratio: float = 0.0,
                 memory_pressure_mb: str | float = None,
                 ondemand_adjust: bool = True,
                 pipeline_load: bool = True,
                 train_memory_over_predict_mb: str | float = None,
                 infer_model_max_idle_ms : Optional[int] = None,
                 has_warmup: bool = False,
                 dump_adjust_info: bool = False, # used for adjust break down
                 profiler_acquire_resource_lock: bool = False, # used for better profiling memory
                 enable_warm_cache_fallback: bool = True, # used for switch and colocate mode
                 profile_gpu_smact: bool = True,
                 profile_gpu_util: bool = True,
                 profile_sm_partition: bool = True,
                 dummy_adjust: bool = False,
                 keep_last_time_stamp: bool = True,
                 max_live_minute: Optional[int] = None) -> None:
        self.mode = mode
        self.use_sta = use_sta
        self.use_sta_train = use_sta_train if use_sta_train is not None else use_sta
        self.cuda_memory_pool_gb = cuda_memory_pool_gb
        self.memory_pool_policy = memory_pool_policy
        self.profile_log = profile_log
        self.server_log = server_log
        self.triton_log = triton_log
        self.mps_log = mps_log
        self.train_profile = train_profile
        self.port = str(int(port) + os.getuid() % 10)
        self.triton_port = str(int(self.port) + 1000)
        self.use_triton = use_triton
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
        self.skip_set_mps_thread_percent = skip_set_mps_thread_percent
        self.dynamic_sm_partition = dynamic_sm_partition
        self.mps_server = None
        self.infer_blob_alloc = infer_blob_alloc
        self.train_mps_thread_percent = train_mps_thread_percent
        self.colocate_skip_malloc = colocate_skip_malloc
        self.colocate_skip_loading = colocate_skip_loading
        self.max_warm_cache_nbytes = max_warm_cache_nbytes
        self.cold_cache_min_capability_nbytes = cold_cache_min_capability_nbytes
        self.cold_cache_max_capability_nbytes = cold_cache_max_capability_nbytes
        self.cold_cache_ratio = cold_cache_ratio
        self.memory_pressure_mb = memory_pressure_mb
        self.ondemand_adjust = ondemand_adjust
        self.pipeline_load = pipeline_load
        self.train_memory_over_predict_mb = train_memory_over_predict_mb
        self.infer_model_max_idle_ms = infer_model_max_idle_ms
        self.has_warmup = has_warmup
        self.dump_adjust_info = dump_adjust_info
        self.profiler_acquire_resource_lock = profiler_acquire_resource_lock
        self.dummy_adjust = dummy_adjust
        self.max_live_minute = max_live_minute
        self.dcgmi_monitor = None
        self.smi_monitor = None
        self.triton_server = None
        if System._last_time_stamp is None or not keep_last_time_stamp:
            self.time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
            System._last_time_stamp = self.time_stamp
        else:
            self.time_stamp = System._last_time_stamp
        self.use_xsched = use_xsched

    def next_time_stamp(self):
        self.time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")

    def launch(self, name: str, subdir: Optional[str] = None, time_stamp:bool=True, 
               infer_model_config: List[InferModelConfig] | InferModelConfig = None,
               dcgmi: bool = False,
               fake_launch: bool = False):
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
        triton_log = f"{self.log_dir}/{self.triton_log}.log"
        mps_log = f"{self.log_dir}/{self.mps_log}.log"
        profile_log = f"{self.log_dir}/{self.profile_log}.log"
        train_profile = f"{self.log_dir}/{self.train_profile}.csv"

        self.cmd_trace = []
        cmd = [
            # "compute-sanitizer", "--tool", "memcheck",
            f"./{get_binary_dir()}/server/colserve", 
            "-p", self.port, 
            "--mode", self.mode, 
            "--use-sta", "1" if self.use_sta else "0", 
            "--use-sta-train", "1" if self.use_sta_train else "0",
            "--profile-log", profile_log
        ]
        if self.cuda_memory_pool_gb is not None:
            cmd += ["--cuda-memory-pool-gb", str(self.cuda_memory_pool_gb)]
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
        if self.use_triton:
            cmd += ['--no-infer', '1']

        # first launch mps
        if self.mps:
            cmd += ["--mps", "1"]
            # self.cmd_trace.append(" ".join([
            #     "sudo", "/opt/mps-control/launch-mps-daemon-private.sh",
            #     "--device", os.environ['CUDA_VISIBLE_DEVICES'], 
            #     "--mps-pipe", os.environ['CUDA_MPS_PIPE_DIRECTORY']
            # ]))
            self.cmd_trace.append(" ".join(CudaMpsCtrl.launch_cmd()))
            if not fake_launch:
                # self.mps_server = subprocess.Popen(
                #     ['sudo', '/opt/mps-control/launch-mps-daemon-private.sh',
                #     '--device', os.environ['CUDA_VISIBLE_DEVICES'], 
                #     '--mps-pipe', os.environ['CUDA_MPS_PIPE_DIRECTORY']],
                #     stderr=subprocess.PIPE, stdout=subprocess.PIPE, env=os.environ.copy())
                with open(mps_log, 'w') as log_file:
                    self.mps_server = subprocess.Popen(
                        CudaMpsCtrl.launch_cmd(),
                        stderr=subprocess.STDOUT, stdout=log_file, 
                        env=os.environ.copy())
                    while True:
                        if self.mps_server.poll() is not None:
                            raise RuntimeError("MPS exited")
                        log_file.flush()
                        with open(mps_log, 'r') as f:
                            content = f.read()
                        if "To connect CUDA applications to this daemon" not in content:
                            time.sleep(0.1)
                        else:
                            print("MPS is ready")
                            break
            else:
                self.mps_server is None
        else:
            cmd += ["--mps", "0"]
            self.mps_server = None

        if self.skip_set_mps_thread_percent:
            cmd += ["--skip-set-mps-thread-percent"]

        if self.infer_blob_alloc:
            cmd += ["--infer-blob-alloc"]

        if self.train_mps_thread_percent is not None:
            cmd += ["--train-mps-thread-percent", 
                    str(self.train_mps_thread_percent)]

        if self.colocate_skip_malloc:
            cmd += ["--colocate-skip-malloc"]
        if self.colocate_skip_loading:
            cmd += ["--colocate-skip-loading"]

        cmd += ['--train-profile', str(train_profile)]
        
        cmd += ['--max-warm-cache-nbytes', str(self.max_warm_cache_nbytes)]
        cmd += ['--cold-cache-min-capability-nbytes', 
                str(self.cold_cache_min_capability_nbytes)]
        cmd += ['--cold-cache-max-capability-nbytes', 
                str(self.cold_cache_max_capability_nbytes)]
        cmd += ['--cold-cache-ratio', str(self.cold_cache_ratio)]

        if self.memory_pressure_mb:
            cmd += ["--memory-pressure-mb", str(self.memory_pressure_mb)]

        if self.ondemand_adjust:
            cmd += ["--ondemand-adjust", "1"]
        else:
            cmd += ["--ondemand-adjust", "0"]

        if self.pipeline_load:
            cmd += ["--pipeline-load", "1"]
        else:
            cmd += ["--pipeline-load", "0"]        

        if self.train_memory_over_predict_mb:
            cmd += ["--train-memory-over-predict-mb", 
                    str(self.train_memory_over_predict_mb)]
        if self.infer_model_max_idle_ms:
            cmd += ["--infer-model-max-idle-ms", 
                    str(self.infer_model_max_idle_ms)]

        if self.has_warmup:
            cmd += ["--has-warmup", "1"]
        else:
            cmd += ["--has-warmup", "0"]

        if self.dump_adjust_info:
            cmd += ["--dump-adjust-info"]

        if self.profiler_acquire_resource_lock:
            cmd += ["--profiler-acquire-resource-lock"]


        if self.dummy_adjust:
            cmd += ["--dummy-adjust"]
        
        if self.use_xsched:
            cmd += ["--use-xsched", "1"]
        else:
            cmd += ["--use-xsched", "0"]

        if self.dynamic_sm_partition:
            cmd += ["--dynamic-sm-partition", "1"]
        else:
            cmd += ["--dynamic-sm-partition", "0"]

        if self.max_live_minute is not None:
            cmd += ["--max-live-minute", str(self.max_live_minute)]

        print("\n---------------------------\n")
        print(" ".join(cmd))

        if fake_launch:
            print(f"  --> fake launch")
            return

        print(f"  --> [server-log] {server_log}  |  [server-profile] {profile_log}")

        if dcgmi:
            with open(f"{self.log_dir}/dcgmi-monitor.log", "w") as log_file:
                self.dcgmi_monitor = subprocess.Popen(
                    ["dcgmi", "dmon", "-e", "1002", 
                    "-i", os.environ['CUDA_VISIBLE_DEVICES']], 
                    stdout=log_file, stderr=subprocess.STDOUT, env=os.environ.copy())

            with open(f'{self.log_dir}/gpu-util.log', 'w') as log_file:
                self.smi_monitor = subprocess.Popen(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv", 
                     "-l", "1", "-i", os.environ['CUDA_VISIBLE_DEVICES']],
                     stdout=log_file, stderr=subprocess.STDOUT, env=os.environ.copy())

        with open(server_log, "w") as log_file:
            env_copy = os.environ.copy()
            if self.mps and self.skip_set_mps_thread_percent:
                print(f'  --> Skip set MPS pct')
            if (not self.skip_set_mps_thread_percent 
                and '_CUDA_MPS_ACTIVE_THREAD_PERCENTAGE' in env_copy):
                env_copy['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = \
                    env_copy['_CUDA_MPS_ACTIVE_THREAD_PERCENTAGE']
                print(f"  --> MPS: {env_copy['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE']}")
                pass
            self.cmd_trace.append(
                f"CUDA_ENV: "
                f"CUDA_VISIBLE_DEVICES {env_copy.get('CUDA_VISIBLE_DEVICES')}"
                f", CUDA_MPS_ACTIVE_THREAD_PERCENTAGE "
                f"{env_copy.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE')}"
            )
            self.cmd_trace.append(" ".join(cmd))
            self.server = subprocess.Popen(cmd, stdout=log_file, 
                                           stderr=subprocess.STDOUT, 
                                           env=env_copy)

        if self.use_triton:
            # with open(triton_log, "w") as log_file:
            with contextlib.nullcontext():
                project_root = os.path.join(os.path.abspath(__file__), 
                                          os.path.pardir, 
                                          os.path.pardir, 
                                          os.path.pardir)
                project_root = os.path.abspath(project_root)
                model_lists = []
                def parse_and_generate_models(input_string: str):
                    # example input
                    """densenet161[38] 
                        path densenet161-b1
                        device cuda
                        batch-size 1
                        num-worker 1
                        max-worker 1"""
                    assert isinstance(input_string, str)
                    model_info = re.search(r'(\w+)\[(\d+)\]', input_string)
                    if model_info:
                        model_name = model_info.group(1)
                        model_quantity = int(model_info.group(2))
                        return [f"{model_name}" if i == 0 else f"{model_name}-{i}" for i in range(model_quantity)]
                    else:
                        return []
                for config in self.infer_model_config:
                    model_lists.extend(parse_and_generate_models(config))
                triton_model_dir = os.path.join(project_root, f'triton_models-{self.port}')
                docker_model_dir = os.path.join('/colsys', f'triton_models-{self.port}')
                print(f'Generate triton model repo for {model_lists}.')
                cp_model(model_lists, get_num_gpu(), triton_model_dir)
                cmd = ['docker', 'run', '-it', '--user', f'{os.getuid()}:{os.getgid()}',
                       '--name', f'colsys-triton-{self.triton_port}',
                       '--rm', '--gpus=all', '--ipc=host',
                       '-p', f'{self.triton_port}:8001',
                       '-v', f'{project_root}:/colsys',
                       '-v', os.environ['HOME'] + ':' + os.environ['HOME'],
                       ]
                if 'STA_RAW_ALLOC_UNIFIED_MEMORY' in os.environ:
                    cmd += ['-v', '/disk2/wyk/tensorrt_backend/install/backends/tensorrt:/opt/tritonserver/backends/tensorrt']
                for key, value in os.environ.items():
                    if key.startswith('CUDA_'):
                        cmd += ['-e', f'{key}={value}']
                cmd += [
                    'nvcr.io/nvidia/tritonserver:23.12-py3',
                    'tritonserver',
                    '--model-load-thread-count=4',
                    f'--model-repository={docker_model_dir}']
                if not 'STA_RAW_ALLOC_UNIFIED_MEMORY' in os.environ:
                    cmd += ['--model-control-mode=explicit']
                self.cmd_trace.append(" ".join(cmd))
                self.triton_server = subprocess.Popen(cmd, stdout=open(triton_log, "w"), stderr=subprocess.STDOUT)
        print('\n')
        with open(f'{self.log_dir}/cmd-trace', 'w') as f:
            f.write("\n\n".join(self.cmd_trace))

        print('Wait ColSys Server start...')
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
        if self.use_triton:
            print('Wait Triton Server start...')
            while True:
                with open(triton_log, 'r') as log_file:
                    if self.triton_server.poll() is not None:
                        print(log_file.read())
                        self.quit_mps()
                        raise RuntimeError("Triton exited")
                    if "Started GRPCInferenceService at" not in log_file.read():
                        time.sleep(0.5)
                    else:
                        break
    
    def stop(self, kill_train: bool = False):
        print("Stop Server")
        if self.server is not None:
            self.server.send_signal(signal.SIGINT)
            self.server = None
        if self.use_triton and self.triton_server is not None:
            self.triton_server.send_signal(signal.SIGINT)
            self.triton_log = None
        if kill_train and self.log_dir is not None:
            train_pids = set()
            print('Force to kill train.')
            with open(f'{self.log_dir}/{self.server_log}.log') as f:
                for line in f:
                    if "[TrainLauncher]: Train TrainJob" in line:
                        regex = r'pid (\d+)'
                        m = re.search(regex, line)
                        if m is not None:
                            pid = int(m.group(1))
                            train_pids.add(pid)
            for pid in train_pids:
                cmd = f'kill -9 {pid}'
                print(f'Execute {cmd}')
                os.system(cmd)
        self.infer_model_config_path = None
        if self.triton_server is not None:
            stop_triton = subprocess.Popen(['docker', 'stop', f'colsys-triton-{self.triton_port}'])
            stop_triton.wait()
        if self.mps_server is not None:
            self.quit_mps()
            self.mps_server.wait()
            self.mps_server = None
        if self.dcgmi_monitor is not None:
            self.dcgmi_monitor.send_signal(signal.SIGINT)
            self.dcgmi_monitor = None
        if self.smi_monitor is not None:
            self.smi_monitor.send_signal(signal.SIGINT)
            self.smi_monitor = None
        if self.log_dir is not None:
            # self.cmd_trace.append(" ".join([
            #     'sudo', '/opt/mps-control/quit-mps-daemon-private.sh',
            #     '--mps-pipe', os.environ['CUDA_MPS_PIPE_DIRECTORY']
            # ]))
            self.cmd_trace.append(" ".join(CudaMpsCtrl.quit_cmd()))
            with open(f'{self.log_dir}/cmd-trace', 'w') as f:
                f.write("\n\n".join(self.cmd_trace))
            self.exit_log_dir = self.log_dir                                                                                                                                   
            self.log_dir = None
    
    def draw_memory_usage(self):
        cmd = (f'python util/profile/memory_trace.py'
               f' -l {self.exit_log_dir}/profile-log.log -o {self.exit_log_dir}')
        print(f'execute {cmd}')
        os.system(cmd)
    
    def draw_trace_cfg(self, time_scale=None):
        cmd = (f'python util/profile/trace_cfg.py'
              f' -t {self.exit_log_dir}/trace-cfg -o {self.exit_log_dir}')
        if time_scale:
            cmd += f' --time-scale {time_scale}'
        print(f'execute {cmd}')
        os.system(cmd)

    def calcuate_train_thpt(self):
        cmd = (f'python util/profile/throughput.py'
               f' --log-dir {self.exit_log_dir}'
               f' > {self.exit_log_dir}/train_thpt 2>&1')
        print(f'execute {cmd}')
        os.system(cmd)

    def draw_infer_slo(self):
        cmd = (f'python util/profile/collect_infer_ltc.py'
               f' -l {self.exit_log_dir}/workload-log'
               f' --slo-output {self.exit_log_dir}')
        print(f'execute {cmd}')
        os.system(cmd)

    def quit_mps(self):
        # quit_mps = subprocess.run([
        #     'sudo', '/opt/mps-control/quit-mps-daemon-private.sh', 
        #     '--mps-pipe', os.environ['CUDA_MPS_PIPE_DIRECTORY']], 
        #     capture_output=True, env=os.environ.copy())
        quit_mps = subprocess.run(CudaMpsCtrl.quit_cmd(), 
                                  capture_output=True, env=os.environ.copy())
        # raise Exception(f"Quit MPS failed: {quit_mps.stderr}")


class HyperWorkload:
    def __init__(self, 
                 concurrency:int, 
                 duration: Optional[int | float] = None, 
                 workload_log:str = "workload-log", 
                 client_log:str = "client-log", 
                 trace_cfg:str = "trace-cfg",
                 infer_timeline:str = "infer-timeline",
                 seed: Optional[int] = None, 
                 warmup: int = 0,
                 wait_warmup_done_sec: float = 0, # see [Note: client timeline] in client/workload/util.h
                 wait_train_setup_sec: float = 0,
                 wait_stable_before_start_profiling_sec: float = 0, 
                 show_result: Optional[int] = None) -> None:
        """
        Args:
            - concurrency: 
                busy loop -> max number of outgoing requests
                workload  -> initial number of outgoing requests
            - duration: workload duration
                for trace workload, duration will be according to the trace
        """
        self.enable_infer = True
        self.enable_train = True
        self.infer_workloads: List[InferWorkloadBase] = []
        self.infer_models: List[InferModel] = []
        self.train_workload: NoneType | TrainWorkload = None
        self.duration = duration
        self.workload_log = workload_log
        self.infer_timeline = infer_timeline
        self.client_log = client_log
        self.trace_cfg = trace_cfg
        self.concurrency = concurrency
        if seed is not None:
            self.seed = seed
        else:
            self.seed = get_global_seed()
        self.warmup = warmup
        self.wait_warmup_done_sec = wait_warmup_done_sec
        self.wait_train_setup_sec = wait_train_setup_sec
        self.wait_stable_before_start_profiling_sec = wait_stable_before_start_profiling_sec
        self.show_result = show_result

        self.infer_extra_infer_sec = self.wait_stable_before_start_profiling_sec

    def set_infer_workloads(self, *infer_workloads: InferWorkloadBase):
        self.infer_workloads = list(infer_workloads)
    
    def set_train_workload(self, train_workload: TrainWorkload | NoneType):
        self.train_workload = train_workload

    def disable_infer(self):
        self.enable_infer = False

    def disable_train(self):
        self.enable_train = False

    def launch_workload(self, server: System, 
                        trace_cfg: Optional[PathLike] = None, 
                        **kwargs):
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

            # record workload summary
            with io.StringIO() as string_io:
                print("WORKLOAD_SUMMARY:", file=string_io)
                for workload in self.infer_workloads:
                    print(f'{workload}', file=string_io)
                    workload.summary_trace(string_io)
                    server.cmd_trace.append(string_io.getvalue())

        self._launch(server, "workload_launcher", infer_trace_args, **kwargs)

    def launch_busy_loop(self, server: System, 
                         infer_models: List[InferModel] = None):
        if infer_models is None:
            infer_models = self.infer_models
        infer_model_args = []
        if len(infer_models) > 0:
            infer_model_args += ["--infer-model"]
            for model in infer_models:
                infer_model_args += [model.model_name]

        self._launch(server, "busy_loop_launcher", infer_model_args)
        
    def _launch(self, server: System, launcher: str, 
                custom_args: List[str] = [], **kwargs):
        assert server is not None
        cmd = [
            # f"./{get_binary_dir()}/client-prefix/src/client-build/{launcher}"
            f"./{get_binary_dir()}/../client/build/{launcher}",
            "-p", server.port,
            "-c", str(self.concurrency),
        ]
        if server.use_triton:
            cmd += ["--triton-port", str(server.triton_port)]
            cmd += ["--triton-config", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../server/triton_models/config.conf"))]
            if 'STA_RAW_ALLOC_UNIFIED_MEMORY' in os.environ:
                cmd += ["--triton-max-memory", "0"]
            else:
                cmd += ["--triton-max-memory", str(int(float(server.cuda_memory_pool_gb) * 1024  * get_num_gpu()))]
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
        if self.warmup > 0 and self.wait_warmup_done_sec > 0:
            cmd += ["--wait-warmup-done-sec", str(self.wait_warmup_done_sec)]

        if self.wait_train_setup_sec > 0:
            cmd += ['--wait-train-setup-sec', str(self.wait_train_setup_sec)]

        if self.wait_stable_before_start_profiling_sec > 0:
            cmd += ['--wait-stable-before-start-profiling-sec', 
                    str(self.wait_stable_before_start_profiling_sec)]

        workload_log = pathlib.Path(server.log_dir) / self.workload_log

        cmd += ['--log', str(workload_log)]
        cmd += ['-v', '1']
        
        infer_timeline = pathlib.Path(server.log_dir) / self.infer_timeline
        cmd += ['--infer-timeline', str(infer_timeline)]

        if self.show_result is not None:
            cmd += ['--show-result', str(self.show_result)]

        cmd += custom_args

        server.cmd_trace.append(" ".join(cmd))
        print(" ".join(cmd))

        if kwargs.get("fake_launch", False):
            print(f"  --> fake launch")
            return

        print(f"  --> [workload-profile] {workload_log}\n")
        with open(f'{server.log_dir}/cmd-trace', 'w') as f:
            f.write("\n\n".join(server.cmd_trace))
        try:
            client_log = pathlib.Path(server.log_dir) / self.client_log
            with open(client_log, "w") as log_file:
                completed = subprocess.run(cmd, stdout=log_file, 
                                           stderr=subprocess.STDOUT, 
                                           env=os.environ.copy())
                if completed.returncode != 0:
                    raise Exception(f"Workload exited with code {completed.returncode}")
        except Exception as e:
            print(f"Workload exited with exception, "
                  f"see detail in {server.log_dir}/{server.server_log}.log "
                  f"and {client_log}")
            server.stop(kill_train=True)
            raise e
