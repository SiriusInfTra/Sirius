import os
import abc
from dataclasses import dataclass
from typing import List, Optional, NamedTuple, Dict
from types import NoneType

import pandas as pd
import numpy as np
from numpy.random import RandomState, MT19937, SeedSequence

from .config import get_global_seed

class InferModel:
    ResNet152 = "resnet152"
    ResNet50 = "resnet50"
    DenseNet161 = "densenet161"
    InceptionV3 = "inception_v3"
    DistilBertBase = "distilbert_base"
    DistilGPT2 = "distilgpt2"
    ViT_b_16 = "vit_b_16"
    ViT_s_16 = "vit_s_16"
    Swin_t = "swin_t"
    EfficientNetV2_s = "efficientnet_v2_s"
    EfficientViT_b2 = "efficientvit_b2"

    model_cnt = 0

    def __init__(self, model_name: str) -> None:
        self.model_id = InferModel.model_cnt
        self.model_name = model_name
        InferModel.model_cnt += 1

    def __hash__(self) -> int:
        return self.model_id
    
    def __repr__(self) -> str:
        return f"InferModel({self.model_id}, {self.model_name})"

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
    
    @classmethod
    def get_multi_model(cls, model_name_list: list[str], total_model: int, num_worker: int):
        num_model_list = [total_model // len(model_name_list) for _ in model_name_list]
        for i in range(total_model % len(model_name_list)):
            num_model_list[i] += 1
        
        client_model_list = []
        server_model_config = []
        for model_name, num_model in zip(model_name_list, num_model_list):
            client_model_list.extend(cls.get_model_list(model_name, num_model))
            server_model_config.append(
                f'''{model_name}[{num_model}]
    path {model_name.lower()}-b1
    device cuda
    batch-size 1
    num-worker {num_worker}
    max-worker 1
                ''')
        return client_model_list, server_model_config

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
            self.seed = get_global_seed()
            self.rs = RandomState(MT19937(SeedSequence(self.seed)))
        else:
            self.seed = seed
            self.rs = RandomState(MT19937(SeedSequence(self.seed)))

    def reset_random_state(self):
        self.rs = RandomState(MT19937(SeedSequence(self.seed)))


class TrainWorkload(NamedTuple):
    train_model: str
    num_epoch: int
    batch_size: int


class AzureInferWorkload(RandomInferWorkload):
    TRACE_D01 = "workload_data/azurefunctions-dataset2019/invocations_per_function_md.anon.d01.csv"

    def __init__(self, 
                 trace_cfg: os.PathLike[str], 
                 interval_sec: float | int, 
                 period_num: int, 
                 func_num: int, 
                 model_list: list[InferModel],
                 period_start_id: int = 0,
                 max_request_sec: Optional[float | int] = None, 
                 avg_request_sec: Optional[float | int] = None, 
                 sort_trace_by: Optional[str] = None,
                 seed: Optional[int] = None) -> None:
        super().__init__(seed)
        self.trace_cfg = trace_cfg
        self.max_request_sec = max_request_sec
        self.avg_request_sec = avg_request_sec
        self.model_list = model_list
        self.interval_sec = interval_sec
        self.period_num = period_num
        self.func_num = func_num
        self.period_start_id = period_start_id
        self.sort_by = sort_trace_by

        if self.max_request_sec is None and self.avg_request_sec is None:
            raise Exception("max_request_sec and avg_request_sec cannot be both None")
        if self.max_request_sec is not None and self.avg_request_sec is not None:
            raise Exception("max_request_sec and avg_request_sec cannot be both specified")

        if self.sort_by is not None and self.sort_by not in ["sum", "var", "num_zero_diff", "var_v2"]:
            raise Exception("sort_by must be either sum, var, num_zero_diff")

        if period_num > 1440:
            raise Exception("period_num must be less than 1440")

    def get_trace(self) -> list[TraceRecord]:
        read_trace_cfg_fn = AzureInferWorkload.read_trace_cfg
        if self.sort_by == "sum":
            read_trace_cfg_fn = AzureInferWorkload.read_top_sum_trace_cfg
        elif self.sort_by == "var":
            read_trace_cfg_fn = AzureInferWorkload.read_top_var_trace_cfg
        elif self.sort_by == "num_zero_diff":
            read_trace_cfg_fn = AzureInferWorkload.read_top_num_zero_diff_trace_cfg
        elif self.sort_by == "var_v2":
            read_trace_cfg_fn = AzureInferWorkload.read_top_var_v2_trace_cfg
        func_freqs = read_trace_cfg_fn(
            self.trace_cfg, self.period_num, self.func_num, 
            period_start_id=self.period_start_id)
        # func_freqs = AzureInferWorkload.read_sorted_trace_cfg(
        #     self.trace_cfg, self.period_num, self.func_num, )
        func_freqs = AzureInferWorkload.normalize_traces(
            func_freqs, 
            max_request_sec=self.max_request_sec, 
            avg_request_sec=self.avg_request_sec)
        trace_list = AzureInferWorkload.convert_traces_record(
            func_freqs, self.interval_sec, self.model_list, self.rs)
        return trace_list

    @classmethod
    def read_top_sum_trace_cfg(cls, trace_cfg: os.PathLike[str], 
                              period_num: int, func_num: int, 
                              period_start_id: int = 0) -> np.ndarray[np.float64]:
        trace_df = pd.read_csv(trace_cfg)
        columns = trace_df.columns[4:]
        trace_df = trace_df[columns[period_start_id: period_start_id + period_num]]
        trace_df['total'] = trace_df.sum(axis=1)
        trace_df = trace_df.sort_values(by='total', ascending=False)
        trace_df = trace_df[columns[period_start_id : period_start_id + period_num]]
        trace_df = trace_df.head(func_num)
        return trace_df.to_numpy()


    @classmethod
    def read_top_var_trace_cfg(cls, trace_cfg: os.PathLike[str], 
                               period_num: int, func_num: int,
                               period_start_id: int = 0) -> np.ndarray[np.float64]:
        trace_df = pd.read_csv(trace_cfg)
        columns = trace_df.columns[4:]
        trace_df = trace_df[columns[period_start_id : period_start_id + period_num]]
        trace_df['var'] = trace_df.var(axis=1)
        trace_df = trace_df.sort_values(by='var', ascending=False)
        trace_df = trace_df[columns[period_start_id : period_start_id + period_num]]
        trace_df = trace_df.head(func_num)
        print(trace_df)
        return trace_df.to_numpy()

    # @classmethod
    # def read_top_num_zero_diff_trace_cfg(
    #         cls, trace_cfg: os.PathLike[str], 
    #         period_num: int, func_num: int,
    #         period_start_id: int = 0
    # ) -> np.ndarray[np.float64]:
    #     trace_df = pd.read_csv(trace_cfg)
    #     columns = trace_df.columns[4:]
    #     trace_df = trace_df[columns[period_start_id : period_start_id + period_num]]
    #     total = trace_df.sum(axis=1)
    #     trace_df = trace_df[total > 0]
    #     zero_threshold = 1
    #     num_none_zeros = np.sum(trace_df > zero_threshold, axis=1)
    #     num_zeros = np.shape(trace_df)[1] - num_none_zeros
    #     trace_df['var'] = np.abs(num_none_zeros - num_zeros)
    #     trace_df = trace_df.sort_values(by='var', ascending=False)
    #     trace_df = trace_df[columns[period_start_id : period_start_id + period_num]]
    #     trace_df = trace_df.head(func_num)
    #     # print(trace_df)
    #     return trace_df.to_numpy()


    @classmethod
    def read_top_var_v2_trace_cfg(
            cls, trace_cfg: os.PathLike[str], 
            period_num: int, func_num: int,
            period_start_id: int = 0
    ) -> np.ndarray[np.float64]:
        trace_df = pd.read_csv(trace_cfg)
        columns = trace_df.columns[4:]
        trace_df = trace_df[columns[period_start_id : period_start_id + period_num]]
        total = trace_df.sum(axis=1)
        trace_df = trace_df[total > max(1, period_num * 0.1)]
        trace_df = trace_df[total < period_num * 10]
        trace_df['var'] = trace_df.var(axis=1)
        trace_df = trace_df.sort_values(by='var', ascending=False)
        trace_df = trace_df[columns[period_start_id : period_start_id + period_num]]
        trace_df = trace_df.head(func_num)
        return trace_df.to_numpy()

    @classmethod
    def read_trace_cfg(cls, trace_cfg: os.PathLike[str], period_num: int, 
                       func_num: int, period_start_id: int = 0) -> np.ndarray[np.float64]:
        func_freq_list = []
        with open(trace_cfg, 'r') as f:
            f.readline() # skip header
            for index, line in enumerate(f):
                if index >= func_num:
                    break
                line = line.strip()
                tokens = line.split(',')
                tokens = tokens[4:] # skip func meta
                tokens = tokens[period_start_id: period_start_id + period_num] # only read period_num data
                freq_data = np.array(tokens, dtype=np.float64)
                func_freq_list.append(freq_data)
        return np.array(func_freq_list, dtype=np.float64)

    @classmethod
    def normalize_traces(cls, func_freqs: np.ndarray[np.float64], 
                         max_request_sec: Optional[float | int] = None,
                         avg_request_sec: Optional[float | int] = None) -> np.ndarray[np.float64]:
        # print(func_freqs)
        assert (max_request_sec is not None and avg_request_sec is None) or \
            (max_request_sec is None) or (avg_request_sec is not None), \
            f"max_request_sec {max_request_sec} avg_request_sec {avg_request_sec}"

        sum_every_sec = np.sum(func_freqs, axis=0)
        if max_request_sec is not None:
            scale_factor = max_request_sec / np.max(sum_every_sec)
        else:
            scale_factor = avg_request_sec / np.mean(sum_every_sec)
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
            trace_record.extend(PoissonInferWorkload.poisson_func_freq(
                poisson_params, infer_model, self.rs))
        return trace_record
    
    @classmethod
    def get_dynamic_poisson_params(cls, simple_poisson_params: dict[InferModel, list[tuple]]) -> list[tuple[InferModel, list[PoissonParam]]]:
        poisson_params = []
        for infer_model, req_dist in simple_poisson_params.items():
            if isinstance(infer_model, str):
                infer_model = InferModel(infer_model)
            if not isinstance(req_dist, list):
                req_dist = [req_dist]
            cur_poisson_param = []
            for start_point, num_req in req_dist:
                if len(cur_poisson_param) == 0 and start_point != 0:
                    cur_poisson_param.append(PoissonParam(0, 0))
                cur_poisson_param.append(PoissonParam(start_point, num_req))
            poisson_params.append((infer_model, cur_poisson_param))
        return poisson_params

class MicrobenchmarkInferWorkload(DynamicPoissonInferWorkload):
    def __init__(self, 
                 model_list: List[InferModel],
                 interval_sec: float | int,
                 max_request_sec: Optional[float | int] = None,
                 fix_request_sec: Optional[float | int] = None,
                 duration: Optional[float | int] = None,
                 period_num: Optional[int] = None,
                 rps_fn = None, # post process rps, Fn(i, rps) -> rps,
                 zipf_alpha: Optional[float] = None,
                 seed: Optional[int] = None) -> None:
        super().__init__(None, None, seed)
        if duration is None and period_num is None:
            raise Exception("duration, period_num and interval_sec cannot be all None")
        if duration is not None and period_num is not None:
            raise Exception("duration and period_num cannot be both specified")
        if max_request_sec is None and fix_request_sec is None:
            raise Exception("max_request_sec and fix_request_sec cannot be both None")
        if max_request_sec is not None and fix_request_sec is not None:
            raise Exception("max_request_sec and fix_request_sec cannot be both specified")
        
        def get_num_request(i):
            if max_request_sec is not None:
                num_request = self.rs.uniform(0, max_request_sec)
            elif fix_request_sec is not None:
                num_request = fix_request_sec
            else:
                raise Exception("max_request_sec and fix_request_sec are None")
            if rps_fn is not None:
                num_request = rps_fn(i, num_request)
            return num_request
        
        if zipf_alpha is not None:
            # do not change self.rs
            tmp_rs = RandomState(MT19937(SeedSequence(self.seed)))
            zipf_seq = tmp_rs.zipf(zipf_alpha, 10000)
            zipf_freq = np.zeros(1 + len(model_list))
            for i in zipf_seq:
                if i <= len(model_list):
                    zipf_freq[i] += 1
            zipf_freq = zipf_freq[1:]
            zipf_freq = zipf_freq / np.sum(zipf_freq)
            zipf_freq = tmp_rs.permutation(zipf_freq)
            print(f'zipf freq: \n', zipf_freq)

        if period_num is None:
            period_num = int(duration / interval_sec + 0.5)
            self.duration = duration
            # self.duration = period_num * interval_sec
        if duration is None:
            self.duration = period_num * interval_sec
        poisson_params = [[] for _ in range(len(model_list))]
        num_model_to_requests = []
        for i in range(period_num):
            # first select a few models to send requests
            num_model = self.rs.randint(1, len(model_list) + 1)
            # num_request = self.rs.uniform(0, max_request_sec)
            # if rps_fn is not None:
            #     num_request = rps_fn(i, num_request)
            num_request = get_num_request(i)
            num_model_to_requests.append(num_model)
            model_req_list = self.rs.choice(np.arange(len(model_list)), num_model, replace=False, 
                                            p = None if zipf_alpha is None else zipf_freq)
            model_num_req = self._split_request(num_request, num_model)
            for model, num_req in zip(model_req_list, model_num_req):
                poisson_params[model].append(PoissonParam(i * interval_sec, num_req))
            for j in range(len(model_list)):
                if j not in set(model_req_list):
                    poisson_params[j].append(PoissonParam(i * interval_sec, 0))
        # print(poisson_params)
        self.poisson_params = []
        for i, poisson_param in enumerate(poisson_params):
            self.poisson_params.append((model_list[i], poisson_param))

        poisson_params_ndarray = np.array(poisson_params)[:, :, 1]
        with np.printoptions(precision=1, suppress=True):
            print('microbenmark total #request: \n', np.array(np.sum(poisson_params_ndarray, axis=0)))
            print('microbenmark request #model : \n', np.array(num_model_to_requests))
            # print(poisson_params_ndarray)

    def _split_request(self, num_request, num_model):
        alpha = np.ones(num_model)
        faction = self.rs.dirichlet(alpha)
        return num_request * faction


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
                # TODO support discontinuous sequence of model id
                assert len(self.infer_workloads) == 1 or index == model.model_id, f"model {model} index {index} not match at {infer_workload}"
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
        
