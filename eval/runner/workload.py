import os
import abc
from dataclasses import dataclass
from typing import List, Optional, NamedTuple, Dict
from types import NoneType

import numpy as np
from numpy.random import RandomState, MT19937, SeedSequence

from .config import get_global_seed

class InferModel:
    model_cnt = 0

    def __init__(self, model_name: str) -> None:
        self.model_id = InferModel.model_cnt
        self.model_name = model_name
        InferModel.model_cnt += 1

    def __hash__(self) -> int:
        return self.model_id
    
    @classmethod
    def reset_model_cnt(cls):
        InferModel.model_cnt = 0

    @classmethod
    def get_model_list(cls, model_name:str, num_model:int):
        if num_model < 1:
            raise Exception("num_model must be greater than 0")
        ret = [InferModel(model_name)]
        for i in range(1, num_model):
            ret.append(InferModel(f"{model_name}-{i}"))
        return ret


class TraceRecord(NamedTuple):
    start_point: float
    model: InferModel


class Workload(abc.ABC):
    @abc.abstractmethod
    def get_params() -> List[TraceRecord]:
        pass 

    def __enter__(self):
        pass
    
    def __exit__(self, type, value, trace):
        pass


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
            self.rs = RandomState(MT19937(SeedSequence(get_global_seed())))
        else:
            self.rs = RandomState(MT19937(SeedSequence(seed)))


class TrainWorkload(NamedTuple):
    train_model: str
    num_epoch: int
    batch_size: int


class AzureInferWorkload(RandomInferWorkload):
    def __init__(self, 
                 trace_cfg: os.PathLike[str], 
                 max_request_sec: float | int, 
                 interval_sec: float | int, 
                 period_num: int, 
                 func_num: int, 
                 model_list: list[InferModel], 
                 seed: Optional[int]) -> None:
        super().__init__(seed)
        self.trace_cfg = trace_cfg
        self.max_request_sec = max_request_sec
        self.model_list = model_list
        self.interval_sec = interval_sec
        self.period_num = period_num
        self.func_num = func_num

    def get_trace(self) -> list[TraceRecord]:
        func_freqs = AzureInferWorkload.read_trace_cfg(
            self.trace_cfg, self.period_num,self.func_num)
        func_freqs = AzureInferWorkload.normalize_traces(
            func_freqs, self.max_request_sec)
        trace_list = AzureInferWorkload.convert_traces_record(
            func_freqs, self.interval_sec, self.model_list, self.rs)
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
        poisson_params = [PoissonParam(interval_sec * period_id, request_per_sec) 
                         for period_id, request_per_sec in enumerate(func_model_freq)] + \
                        [PoissonParam(len(func_model_freq) * interval_sec, 0)]
        return PoissonInferWorkload.poisson_func_freq(poisson_params, model, rs)
    
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


class PoissonInferWorkload(RandomInferWorkload):
    def __init__(self, 
                 poisson_params: list[tuple[InferModel, PoissonParam]], 
                 duration: float | int, 
                 seed: Optional[int] = None) -> None:
        super().__init__(seed)
        self.poisson_params = poisson_params
        self.duration = duration

    @classmethod
    def poisson_func_freq(cls, poisson_params: list[PoissonParam], model: InferModel, 
                          rs: RandomState) -> list[TraceRecord]:
        assert poisson_params[-1].request_per_sec == 0
        period_id = 0
        next_duration_norm = rs.exponential()
        start_point = 0
        trace_list = []
        while period_id < len(poisson_params) - 1:
            if poisson_params[period_id].request_per_sec == 0:
                start_point = poisson_params[period_id + 1].time_point
                period_id = period_id + 1
                continue
            next_duration_local = next_duration_norm / poisson_params[period_id].request_per_sec
            start_point_unsafe = start_point + next_duration_local
            next_time_point = float(poisson_params[period_id + 1].time_point)
            if start_point_unsafe < next_time_point:
                start_point = start_point_unsafe
                trace_list.append(TraceRecord(start_point, model))
                next_duration_norm = rs.exponential()
            else:
                next_duration_norm -= (next_time_point - start_point) * poisson_params[period_id].request_per_sec
                start_point = next_time_point
                period_id = period_id + 1
        return trace_list

    def get_trace(self) -> list[TraceRecord]:
        trace_record: list[TraceRecord] = []
        for infer_model, poisson_param in self.poisson_params:
            assert poisson_param.time_point == 0
            poisson_params = [poisson_param, PoissonParam(self.duration, 0)]
            trace_record.extend(PoissonInferWorkload.poisson_func_freq(
                poisson_params, infer_model, self.rs))
        return trace_record


class DynamicPoissonInferWorkload(RandomInferWorkload):
    def __init__(self, 
                 poisson_params: list[tuple[InferModel, list[PoissonParam]]], 
                 duration: int | float, 
                 seed: Optional[int] = None) -> None:
        super().__init__(seed)
        self.poisson_params = poisson_params
        self.duration = duration
    
    def get_trace(self) -> list[TraceRecord]:
        trace_record: list[TraceRecord] = []
        for infer_model, poisson_params in self.poisson_params:
            poisson_params = poisson_params + [PoissonParam(self.duration, 0)]
            trace_record.append(PoissonInferWorkload.poisson_func_freq(
                poisson_params, infer_model, self.rs))
        return trace_record


class InferTraceDumper:
    def __init__(self, 
                 infer_workloads: list[InferWorkloadBase], 
                 trace_cfg: os.PathLike) -> None:
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
        
