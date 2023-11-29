
from __future__ import annotations
import datetime
import time
import numpy as np
from numpy.random import RandomState, MT19937, SeedSequence
import abc
import os
from os import PathLike
import pathlib
import subprocess
import pynvml
from dataclasses import dataclass

from types import NoneType
from typing import Optional, NamedTuple, Dict


GPU_UUIDs = []
pynvml.nvmlInit()
for i in range(pynvml.nvmlDeviceGetCount()):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    uuid_as_str = pynvml.nvmlDeviceGetUUID(handle)
    if not isinstance(uuid_as_str, str):
        uuid_as_str = uuid_as_str.decode()
    GPU_UUIDs.append(uuid_as_str)
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
if 'https_proxy' in os.environ:
    del os.environ['https_proxy']


class System:
    class ServerMode:
        Normal = "normal"
        ColocateL1 = "colocate-l1"
        ColocateL2 = "colocate-l2"

    def __init__(self, mode:str, use_sta:bool, cuda_memory_pool_gb:str,
                 profile_log:str, server_log:str, port:str= "18080",
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

    def __init__(self, workload_log: str, client_log: str, trace_cfg: str, concurrency: int, duration: Optional[int | float] = None,
                  seed: Optional[int] = None, delay_before_infer: float = 0, warmup: Optional[int] = None) -> None:
        self.infer_workloads: list[InferWorkloadBase] = []
        self.train_workload: NoneType | TrainWorkload = None
        self.duration = duration
        self.workload_log = workload_log
        self.client_log = client_log
        self.trace_cfg = trace_cfg
        self.concurrency = concurrency
        self.seed = seed
        self.delay_before_infer = delay_before_infer
        self.warmup = warmup

    def dump_trace(self, trace_cfg: PathLike):
        InferTraceDumper(self.infer_workloads, trace_cfg).dump()
    
    def launch(self, server: System, trace_cfg: Optional[PathLike] = None):
        assert server.server is not None
        cmd = [
            "./build/hybrid_workload",
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
            self.dump_trace(trace_cfg)
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
        
        if self.warmup is not None:
            cmd += ["--warmup", str(self.warmup)]

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


class Workload(abc.ABC):
    @abc.abstractmethod
    def get_params() -> list[TraceRecord]:
        pass 

    def __enter__(self):
        pass
    
    def __exit__(self, type, value, trace):
        pass


@dataclass
class InferModel:
    model_id: int
    model_name: str

    def __hash__(self) -> int:
        return self.model_id


class TraceRecord(NamedTuple):
    start_point: float
    model: InferModel


class PoissonParam(NamedTuple):
    time_point: float | int
    request_per_sec: float | int


class InferWorkloadBase(abc.ABC):
    @abc.abstractmethod
    def get_trace(self) -> list[TraceRecord]:
        pass


class RandomInferWorkload(InferWorkloadBase):
    def __init__(self, seed: Optional[int] = None) -> None:
        super().__init__()
        if seed is None:
            self.rs = RandomState()
        else:
            self.rs = RandomState(MT19937(SeedSequence(seed)))


class TrainWorkload(NamedTuple):
    train_model: str
    num_epoch: int
    batch_size: int


class AzureInferWorkload(RandomInferWorkload):
    def __init__(self, trace_cfg: os.PathLike[str], max_request_sec: float | int, 
                 interval_sec: float | int, period_num: int, func_num: int, 
                 model_list: list[InferModel], seed: Optional[int]) -> None:
        super().__init__(seed)
        self.trace_cfg = trace_cfg
        self.max_request_sec = max_request_sec
        self.model_list = model_list
        self.interval_sec = interval_sec
        self.period_num = period_num
        self.func_num = func_num

    def get_trace(self) -> list[TraceRecord]:
        func_freqs = AzureInferWorkload.read_trace_cfg(self.trace_cfg, self.period_num, self.func_num)
        func_freqs = AzureInferWorkload.normalize_traces(func_freqs, self.max_request_sec)
        trace_list = AzureInferWorkload.convert_traces_record(func_freqs, self.interval_sec, self.model_list, self.rs)
        return trace_list

    @classmethod
    def read_trace_cfg(cls, trace_cfg: os.PathLike[str], period_num: int, func_num: int) -> np.ndarray[np.float64]:
        func_freq_list = []
        with open(trace_cfg, 'r') as f:
            f.readline() # skip header
            for index, line in enumerate(f):
                if index >= func_num:
                    break
                line = line.strip()
                tokens = line.split(',')
                tokens = tokens[4:] # skip func meta
                tokens = tokens[:period_num] # only read period_num data
                freq_data = np.array(tokens, dtype=np.float64)
                func_freq_list.append(freq_data)
        return np.array(func_freq_list, dtype=np.float64)

    @classmethod
    def normalize_traces(cls, func_freqs: np.ndarray[np.float64], max_request_sec: float | int) -> np.ndarray[np.float64]:
        # print(func_freqs)
        sum_every_sec = np.sum(func_freqs, axis=0)
        scale_factor = max_request_sec / np.max(sum_every_sec)
        # print(f"scale_factor={scale_factor}")
        func_traces_normalized = func_freqs * scale_factor
        # print(func_traces_normalized)
        return func_traces_normalized
    
    @classmethod
    def poisson_func_freq(cls, func_model_freq: np.ndarray[np.float64], 
                          interval_sec: float | int, model: InferModel, 
                          rs: RandomState) -> list[TraceRecord]:
        poisson_parms = [PoissonParam(interval_sec * period_id, request_per_sec) 
                         for period_id, request_per_sec in enumerate(func_model_freq)] + \
                        [PoissonParam(len(func_model_freq) * interval_sec, 0)]
        return PoissonInferWorkload.poisson_func_freq(poisson_parms, model, rs)
    
    @classmethod
    def convert_traces_record(cls, func_freqs: np.ndarray[np.float64], interval_sec: float | int, 
                              model_list: list[InferModel], rs: RandomState) -> np.ndarray[np.float64]:
        trace_list = []
        func_model_freqs = np.zeros((len(model_list), np.shape(func_freqs)[1]), np.float64)
        
        for index, func_freq in enumerate(func_freqs):
            func_model_freqs[index % len(model_list)] += func_freq
        
        for model, func_freq in zip(model_list, func_model_freqs):
            # print(f"func_freq={func_freq}")
            trace_list += cls.poisson_func_freq(func_freq, interval_sec, model, rs)
            # break
        # print("\n".join(["%.2f" % trace.start_point for trace in trace_list]))

        return trace_list


class InferTraceDumper:
    def __init__(self, infer_workloads: list[InferWorkloadBase], trace_cfg: PathLike) -> None:
        self.infer_workloads = infer_workloads
        self.trace_cfg = trace_cfg
    
    def dump(self) -> None:
        model_list: list[InferModel] = []
        trace_list: list[TraceRecord] = []
        for infer_workload in self.infer_workloads:
            model_set_local: set[InferModel] = set()
            trace_list_local = infer_workload.get_trace()
            for trace in trace_list_local:
                model_set_local.add(trace.model)
            model_list_local: list[InferModel] = sorted(list(model_set_local), 
                                                        key=lambda model: model.model_id)
            # check trace and update trace
            for index, model in enumerate(model_list_local):
                assert index == model.model_id, f"model index not match at {infer_workload}."
                model.model_id += len(trace_list)
            trace_list.extend(trace_list_local)
            model_list.extend(model_list_local)
        trace_list.sort(key=lambda trace: trace.start_point)
        with open(self.trace_cfg, 'w') as f:
            f.write("# model_id,model_name\n")
            for infer_model in model_list:
                f.write(f"{infer_model.model_id},{infer_model.model_name}\n")
            f.write("# start_point,model_id\n")
            for trace in trace_list:
                f.write(f"{'%.4f' % trace.start_point},{trace.model.model_id}\n")
        

class PoissonInferWorkload(RandomInferWorkload):
    def __init__(self, poission_parms: list[tuple[InferModel, PoissonParam]], 
                 duration: float | int, seed: Optional[int] = None) -> None:
        super().__init__(seed)
        self.poission_parms = poission_parms
        self.duration = duration

    @classmethod
    def poisson_func_freq(cls, poission_parms: list[PoissonParam], model: InferModel, 
                          rs: RandomState) -> list[TraceRecord]:
        assert poission_parms[-1].request_per_sec == 0
        period_id = 0
        next_duration_norm = rs.exponential()
        start_point = 0
        trace_list = []
        while period_id < len(poission_parms) - 1:
            if poission_parms[period_id].request_per_sec == 0:
                start_point = poission_parms[period_id + 1].time_point
                period_id = period_id + 1
                continue
            next_duration_local = next_duration_norm / poission_parms[period_id].request_per_sec
            start_point_unsafe = start_point + next_duration_local
            next_time_point = float(poission_parms[period_id + 1].time_point)
            if start_point_unsafe < next_time_point:
                start_point = start_point_unsafe
                trace_list.append(TraceRecord(start_point, model))
                next_duration_norm = rs.exponential()
            else:
                next_duration_norm -= (next_time_point - start_point) * poission_parms[period_id].request_per_sec
                start_point = next_time_point
                period_id = period_id + 1
        return trace_list

    def get_trace(self) -> list[TraceRecord]:
        trace_record: list[TraceRecord] = []
        for infer_model, poisson_parm in self.poission_parms:
            assert poisson_parm.time_point == 0
            poisson_parms = [poisson_parm, PoissonParam(self.duration, 0)]
            trace_record.append(PoissonInferWorkload.poisson_func_freq(poisson_parms, infer_model, self.rs))
        return trace_record


class DynamicPoissonInferWorkload(RandomInferWorkload):
    def __init__(self, possion_parms: list[tuple[InferModel, list[PoissonParam]]], 
                 duration: int | float, seed: Optional[int] = None) -> None:
        super().__init__(seed)
        self.poisson_parms = possion_parms
        self.duration = duration
    
    def get_trace(self) -> list[TraceRecord]:
        trace_record: list[TraceRecord] = []
        for infer_model, poisson_parms in self.poisson_parms:
            poisson_parms = poisson_parms + [PoissonParam(self.duration, 0)]
            trace_record.append(PoissonInferWorkload.poisson_func_freq(poisson_parms, infer_model, self.rs))
        return trace_record


print('CUDA_VISIBLE_DEVICES={}, CUDA_MPS_PIPE_DIRECTORY={}'.format(
    os.environ['CUDA_VISIBLE_DEVICES'], os.environ['CUDA_MPS_PIPE_DIRECTORY']))