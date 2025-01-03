import os
import argparse
from dataclasses import dataclass

from runner import *
import workload_collections as wkld_coll

# !!! do not import * from run_comm

use_time_stamp = True
retry_if_fail = False
skip_fail = False

enable_uniform = False
enable_skewed = False

retry_limit = 1

# for debug
fake_launch = False

def get_unique_port():
    cuda_device_env = os.environ['CUDA_VISIBLE_DEVICES']
    # assert ',' not in cuda_device
    cuda_device = cuda_device_env.split(',')[0]
    try:
        cuda_device = int(cuda_device)
    except:
        cuda_device = GPU_UUIDs.index(cuda_device)
    port = 18100
    port += cuda_device
    return port

## MARK: Configurations
## =========================================================== ##

@dataclass
class HighLoad:
    rps: int = 50
    mps_infer: int = 30
    mps_train: int = 70
    enable: bool = True

@dataclass
class LowLoad:
    rps: int = 5
    mps_infer: int = 30
    mps_train: int = 70
    enable: bool = True

@dataclass
class HybridLoad:
    high_rps = HighLoad.rps
    low_rps = LowLoad.rps
    mps_infer: int = HighLoad.rps
    mps_train: int = HighLoad.rps
    enable: bool = True


# MARK: Trace Config
class UniformConfig:
    train_model = 'swin_b'
    train_batch_size = 72
    train_global_batch_size = 500 # not used, hard code for global batch size and dataset size
    train_dataset_size = 1000 
    train_epoch_time = 5.5 # used for predict number epoch

    model_list = [InferModel.DenseNet161, InferModel.EfficientNetV2_s, 
                  InferModel.EfficientViT_b2, InferModel.DistilBertBase, 
                  InferModel.ResNet152, InferModel.DistilGPT2] 
    num_model = 64
    interval_sec = 20
    duration = 120
    port = str(get_unique_port())
    enable = enable_uniform

    low_load = LowLoad(enable=True)
    high_load = HighLoad(enable=False)
    hybrid_load = HybridLoad(enable=False)


class UniformConfig_v2:
    train_model = 'swin_b'
    train_batch_size = 72
    train_global_batch_size = 500 # not used, hard code for global batch size and dataset size
    train_dataset_size = 1000 
    train_epoch_time = 3 # used for predict number epoch
    real_data = False

    model_list = [InferModel.DenseNet161, InferModel.EfficientNetV2_s, 
                  InferModel.EfficientViT_b2, InferModel.DistilBertBase, 
                  InferModel.ResNet152, InferModel.DistilGPT2] 
    # num_model = 64
    # num_model = 60 * get_num_gpu()
    num_model = runner.scale_up_by_num_gpu(56)
    interval_sec = 20  # 10/20 sec seem to be good choice
    duration = 300
    port = str(get_unique_port())
    enable = enable_uniform


class SkewedConfig:
    # train_model = 'gpt2'
    # train_batch_size = 20
    # train_global_batch_size = 250
    # train_dataset_size = 500
    # train_epoch_time = 5
    train_model = 'swin_b'
    train_batch_size = 72
    train_global_batch_size = 500 # not used, hard code for global batch size and dataset size
    train_dataset_size = 1000 
    train_epoch_time = 5.5 # used for predict number epoch

    model_list = [InferModel.DenseNet161, InferModel.EfficientNetV2_s, 
                  InferModel.EfficientViT_b2, InferModel.DistilBertBase, 
                  InferModel.ResNet152, InferModel.DistilGPT2] 
    num_model = 64
    interval_sec = 20
    duration = 120
    zipf_aplha = 1.2 # large alpha -> more skewed
    port = str(get_unique_port())
    enable = enable_skewed

    low_load = LowLoad(enable=True)
    high_load = HighLoad(enable=True)
    hybrid_load = HybridLoad(enable=False)


class SkewedConfig_v2:
    train_model = 'swin_b'
    train_batch_size = 72
    train_global_batch_size = 500
    train_dataset_size = 1000
    train_epoch_time = 3

    model_list = [InferModel.DenseNet161, InferModel.EfficientNetV2_s, 
                  InferModel.EfficientViT_b2, InferModel.DistilBertBase, 
                  InferModel.ResNet152, InferModel.DistilGPT2] 
    # num_model = 64
    num_model = runner.scale_up_by_num_gpu(56)
    interval_sec = 20
    duration = 300
    port = str(get_unique_port())
    enable = enable_skewed


class LLMWorkloadConfig:
    train_model = 'swin_b'
    train_batch_size = 72
    train_global_batch_size = 500
    train_dataset_size = 1000
    train_epoch_time = 3

    duration = 300
    port = str(get_unique_port())


# MARK: Workload
## =========================================================== ##

def get_train_epoch(train_epoch_time, duration):
    return int(duration / train_epoch_time + 10)


def _workload_v2(wkld_type, client_model_list, infer_only,
                train_model:str, train_epoch:Optional[int], 
                train_batch_size:int, train_epoch_time:float,
                interval_sec:int, duration:int):
    workload = HyperWorkload(concurrency=2048,
                             warmup=5,
                             wait_warmup_done_sec=5,
                             wait_train_setup_sec=60 ,
                             wait_stable_before_start_profiling_sec=10)
    InferModel.reset_model_cnt()
    if not infer_only:
        if train_epoch is None:
            train_epoch = get_train_epoch(train_epoch_time, duration)
        print(f'Train {train_model} Epoch {train_epoch} Batch {train_batch_size}')
        workload.set_train_workload(
            train_workload=TrainWorkload(train_model, train_epoch, train_batch_size))
    if isinstance(wkld_type, str):
        wkld_type = getattr(wkld_coll, wkld_type)
    workload.set_infer_workloads(wkld_type(
        model_list=client_model_list,
        interval_sec=interval_sec,
        duration=duration + workload.infer_extra_infer_sec))
    return workload


def uniform(rps, client_model_list, infer_only=True, rps_fn=None,
            train_model:str = UniformConfig.train_model, 
            train_epoch:Optional[int] = None, 
            train_batch_size:int = UniformConfig.train_batch_size):
    # assert train_epoch is not None, "train_epoch is None"
    if train_epoch is None:
        train_epoch = get_train_epoch(UniformConfig.train_epoch_time, UniformConfig.duration)
    print(f'Train {train_model} Epoch {train_epoch} Batch {train_batch_size}')
    workload = HyperWorkload(concurrency=2048,
                             warmup=5,
                             wait_warmup_done_sec=5,
                             wait_train_setup_sec=40 ,
                             wait_stable_before_start_profiling_sec=10)
    InferModel.reset_model_cnt()
    if not infer_only:
        workload.set_train_workload(
            train_workload=TrainWorkload(train_model, train_epoch, train_batch_size))
    workload.set_infer_workloads(MicrobenchmarkInferWorkload_v1(
        model_list=client_model_list,
        interval_sec=UniformConfig.interval_sec, fix_request_sec=rps, rps_fn=rps_fn,
        duration=UniformConfig.duration + workload.infer_extra_infer_sec,
    ))
    return workload


def uniform_v2(wkld_type, client_model_list, infer_only=True, 
               train_model:Optional[str] = None,
               train_epoch:Optional[int] = None,
               train_batch_size:Optional[int] = None,
               train_epoch_time:Optional[float] = None):
    if train_model is None:
        train_model = UniformConfig_v2.train_model
    if train_batch_size is None:
        train_batch_size = UniformConfig_v2.train_batch_size
    if train_epoch_time is None:
        train_epoch_time = UniformConfig_v2.train_epoch_time
    return _workload_v2(wkld_type, client_model_list, infer_only,
                        train_model, train_epoch, 
                        train_batch_size, train_epoch_time,
                        UniformConfig_v2.interval_sec, 
                        UniformConfig_v2.duration)

    # workload = HyperWorkload(concurrency=2048,
    #                          warmup=5,
    #                          wait_warmup_done_sec=5,
    #                          wait_train_setup_sec=40 ,
    #                          wait_stable_before_start_profiling_sec=10)
    # InferModel.reset_model_cnt()
    # if not infer_only:
    #     if train_epoch is None:
    #         train_epoch = get_train_epoch(train_epoch_time, 
    #                                       UniformConfig_v2.duration)
    #         # train_batch_size = UniformConfig_v2.train_batch_size
    #     print(f'Train {train_model} Epoch {train_epoch} Batch {train_batch_size}')
    #     workload.set_train_workload(
    #         train_workload=TrainWorkload(train_model, train_epoch, train_batch_size))
    # if isinstance(wkld_type, str):
    #     wkld_type = getattr(wkld_coll, wkld_type)
    # workload.set_infer_workloads(wkld_type(
    #     model_list=client_model_list,
    #     interval_sec=UniformConfig_v2.interval_sec,
    #     duration=UniformConfig_v2.duration + workload.infer_extra_infer_sec))
    # return workload


def skewed(rps, client_model_list, infer_only=True, rps_fn=None,
           train_model:str =SkewedConfig.train_model, 
           train_epoch:Optional[int] = None, 
           train_batch_size:int = SkewedConfig.train_batch_size):
    # assert train_epoch is not None, "train_epoch is None"
    if train_epoch is None:
        train_epoch = get_train_epoch(SkewedConfig.train_epoch_time, SkewedConfig.duration)
    workload = HyperWorkload(concurrency=2048,
                             warmup=5,
                             wait_warmup_done_sec=5,
                             wait_train_setup_sec=40,
                             wait_stable_before_start_profiling_sec=10)
    InferModel.reset_model_cnt()
    if not infer_only:
        workload.set_train_workload(
            train_workload=TrainWorkload(train_model, train_epoch, train_batch_size))
    workload.set_infer_workloads(MicrobenchmarkInferWorkload_v1(
        model_list=client_model_list,
        interval_sec=SkewedConfig.interval_sec, fix_request_sec=rps, rps_fn=rps_fn,
        zipf_alpha=SkewedConfig.zipf_aplha,
        duration=SkewedConfig.duration + workload.infer_extra_infer_sec,
    ))
    return workload


def skewed_v2(wkld_type, client_model_list, infer_only=True,
              train_model:Optional[str] = None,
              train_epoch:Optional[int] = None,
              train_batch_size:Optional[int] = None,
              train_epoch_time:Optional[float] = None):
    if train_model is None:
        train_model = SkewedConfig_v2.train_model
    if train_batch_size is None:
        train_batch_size = SkewedConfig_v2.train_batch_size
    if train_epoch_time is None:
        train_epoch_time = SkewedConfig_v2.train_epoch_time
    return _workload_v2(wkld_type, client_model_list, infer_only,
                        train_model, train_epoch, 
                        train_batch_size, train_epoch_time,
                        SkewedConfig_v2.interval_sec, 
                        SkewedConfig_v2.duration)
    

def burstgpt(infer_only=True, 
        train_model:Optional[str] = None,
        train_epoch:Optional[int] = None,
        train_batch_size:Optional[int] = None,
        train_epoch_time:Optional[float] = None):
    if train_model is None:
        train_model = LLMWorkloadConfig.train_model
    if train_batch_size is None:
        train_batch_size = LLMWorkloadConfig.train_batch_size
    if train_epoch_time is None:
        train_batch_size = LLMWorkloadConfig.train_epoch_time
    workload = HyperWorkload(concurrency=2048,
                             warmup=5,
                             wait_warmup_done_sec=5,
                             wait_train_setup_sec=60,
                             wait_stable_before_start_profiling_sec=10,
                             is_llm_workload=True)
    if not infer_only:
        if train_epoch is None:
            train_epoch = get_train_epoch(train_epoch_time, 
                                          LLMWorkloadConfig.duration)
        workload.set_train_workload(
            train_workload=TrainWorkload(train_model, train_epoch, train_batch_size))
    workload.set_infer_workloads(LLMInferWorkload())
    return workload


def _run(system: System, workload: HyperWorkload, server_model_config: str, unit: str, tag: str):
    try:
        system.launch(unit, tag, time_stamp=use_time_stamp,
                    infer_model_config=server_model_config, fake_launch=fake_launch)
        workload.launch_workload(system, fake_launch=fake_launch)
        system.stop()
    except Exception as e:
        print(f"Failed to run {unit} {tag}: {e}")
        if retry_if_fail:
            for retry_cnt in range(retry_limit):
                print(f"\n\x1b[33;1m### Retry [{unit} {tag}] x {retry_cnt} ###\x1b[0m")
                try:
                    time.sleep(3)
                    system.launch(unit, f'{tag}-retry-{retry_cnt}', time_stamp=use_time_stamp,
                            infer_model_config=server_model_config)
                    workload.launch_workload(system)
                    system.stop()
                except Exception as e:
                    print(f"Failed to run {unit} {tag}: {e}")
                    if retry_cnt == retry_limit - 1:
                        raise e
                else:
                    break
        else:
            raise e
    if not fake_launch:
        time.sleep(5)
    else:
        time.sleep(1)

    system.draw_trace_cfg()
    if not fake_launch:
        system.draw_memory_usage()
        system.draw_infer_slo()
        system.calcuate_train_thpt()
    

def run(system: System, workload: HyperWorkload, 
        server_model_config: str, unit: str, tag: str):
    print(f'\x1b[32;1m[{unit} {tag}]\x1b[0m')
    if skip_fail:
        try:
            _run(system, workload, server_model_config, unit, tag)
        except Exception as e:
            print(f"\n\x1b[33;1m### Skip {unit} {tag}: {e} ###\x1b[0m")
            time.sleep(1)
    else:
        _run(system, workload, server_model_config, unit, tag)
    print("\n===========================\n===========================\n")



