import os
import argparse
from runner import *
from dataclasses import dataclass

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
    cuda_device = os.environ['CUDA_VISIBLE_DEVICES']
    assert ',' not in cuda_device
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


# MARK: Workload
## =========================================================== ##

def get_train_epoch(train_epoch_time, duration):
    return int(duration / train_epoch_time + 5)

def uniform(rps, client_model_list, infer_only=True, rps_fn=None,
            train_model:str = UniformConfig.train_model, 
            train_epoch:int = None, 
            train_batch_size:int = UniformConfig.train_batch_size):
    # assert train_epoch is not None, "train_epoch is None"
    if train_epoch is None:
        train_epoch = get_train_epoch(UniformConfig.train_epoch_time, UniformConfig.duration)
    print(f'Train {train_model} Epoch {train_epoch} Batc {train_batch_size}')
    workload = HyperWorkload(concurrency=2048,
                             warmup=5,
                             wait_warmup_done_sec=5,
                             wait_train_setup_sec=40 ,
                             wait_stable_before_start_profiling_sec=10)
    InferModel.reset_model_cnt()
    if not infer_only:
        workload.set_train_workload(
            train_workload=TrainWorkload(train_model, train_epoch, train_batch_size))
    workload.set_infer_workloads(MicrobenchmarkInferWorkload(
        model_list=client_model_list,
        interval_sec=UniformConfig.interval_sec, fix_request_sec=rps, rps_fn=rps_fn,
        duration=UniformConfig.duration + workload.infer_extra_infer_sec,
    ))
    return workload


def skewed(rps, client_model_list, infer_only=True, rps_fn=None,
           train_model:str =SkewedConfig.train_model, 
           train_epoch:int = None, 
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
    workload.set_infer_workloads(MicrobenchmarkInferWorkload(
        model_list=client_model_list,
        interval_sec=SkewedConfig.interval_sec, fix_request_sec=rps, rps_fn=rps_fn,
        zipf_alpha=SkewedConfig.zipf_aplha,
        duration=SkewedConfig.duration + workload.infer_extra_infer_sec,
    ))
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
    time.sleep(5)
    system.draw_memory_usage()
    system.draw_trace_cfg()
    system.calcuate_train_thpt()
    

def run(system: System, workload: HyperWorkload, server_model_config: str, unit: str, tag: str):
    print(f'\x1b[32;1m[{unit} {tag}]\x1b[0m')
    if skip_fail:
        try:
            _run(system, workload, server_model_config, unit, tag)
        except Exception as e:
            print(f"\n\x1b[33;1m### Skip {unit} {tag}: {e} ###\x1b[0m")
            time.sleep(1)
    else:
        _run(system, workload, server_model_config, unit, tag)


