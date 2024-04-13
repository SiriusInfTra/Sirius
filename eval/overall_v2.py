import os
import argparse
from runner import *
from dataclasses import dataclass

set_global_seed(42)

use_time_stamp = True
retry_if_fail = False
skip_fail = False

run_colsys  = False
run_um_mps = False
run_task_switch = False
run_static_partition = False
run_static_partition_I = False
run_static_partition_F = False
run_infer_only = False

enable_uniform = False
enable_skewed = False
enable_azure = False
enable_azure_profile_memory = False
enable_hybrid = False

# should be false to eval infer-only
infer_only_without_mps = False

# args parser
parser = argparse.ArgumentParser()
parser.add_argument('--colsys', action='store_true')
parser.add_argument('--um-mps', action='store_true')
parser.add_argument('--task-switch', action='store_true')
parser.add_argument('--static-partition', action='store_true')
parser.add_argument('--static-partition-i', action='store_true')
parser.add_argument('--static-partition-f', action='store_true')
parser.add_argument('--infer-only', action='store_true')
parser.add_argument('--uniform', action='store_true')
parser.add_argument('--skewed', action='store_true')
parser.add_argument('--hybrid', action='store_true')
parser.add_argument('--strawman', action='store_true')
parser.add_argument('--azure', action='store_true')
parser.add_argument('--azure-profile-memory', action='store_true')
parser.add_argument('--all-sys', action='store_true')
parser.add_argument('--all-workload', action='store_true')
parser.add_argument('--infer-only-without-mps', action='store_true')
args = parser.parse_args()

if args.colsys or args.all_sys:
    run_colsys = True
if args.um_mps or args.all_sys:
    run_um_mps = True
if args.task_switch or args.all_sys:
    run_task_switch = True
if args.static_partition or args.all_sys:
    run_static_partition = True
    run_static_partition_I = True
    run_static_partition_F = True
if args.static_partition_i or args.all_sys:
    run_static_partition = True
    run_static_partition_I = True
if args.static_partition_f or args.all_sys:
    run_static_partition = True
    run_static_partition_F = True
if args.infer_only or args.all_sys:
    run_infer_only = True

if args.uniform or args.all_workload:
    enable_uniform = True
if args.skewed or args.all_workload:
    enable_skewed = True
if args.azure or args.all_workload:
    enable_azure = True
if args.azure_profile_memory or args.all_workload:
    enable_azure_profile_memory = True
if args.hybrid or args.all_workload:
    enable_hybrid = True

if args.infer_only_without_mps:
    infer_only_without_mps = True

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
    mps_infer: int = HighLoad.mps_infer
    mps_train: int = HighLoad.mps_train
    enable: bool = enable_hybrid

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

    low_load = LowLoad(enable=False)
    high_load = HighLoad(enable=True)
    hybrid_load = HybridLoad(enable=False)


class SkewedConfig:
    # train_model = 'gpt2'
    # train_batch_size = 20
    # train_global_batch_size = 250
    # train_dataset_size = 500
    # train_epoch_time = 5.5
    train_model = 'swin_b'
    train_batch_size = 72
    train_global_batch_size = 500 # not used, hard code for global batch size and dataset size
    train_dataset_size = 1000 
    train_epoch_time = 5 # used for predict number epoch

    model_list = [InferModel.DenseNet161, InferModel.EfficientNetV2_s, 
                  InferModel.EfficientViT_b2, InferModel.DistilBertBase, 
                  InferModel.ResNet152, InferModel.DistilGPT2] 
    num_model = 64
    interval_sec = 20
    duration = 120
    zipf_aplha = 1.05 # large alpha -> more skewed
    port = str(get_unique_port())
    enable = enable_skewed

    low_load = LowLoad(enable=True)
    high_load = HighLoad(enable=True)
    hybrid_load = HybridLoad(enable=False)


class AzureConfig:
    train_model = 'swin_b'
    train_batch_size = 72

    model_list = [InferModel.DenseNet161, InferModel.EfficientNetV2_s, 
                  InferModel.EfficientViT_b2, InferModel.DistilBertBase, 
                  InferModel.ResNet152, InferModel.DistilGPT2] 
    num_model = 64
    interval_sec = 5
    duration = 180 
    period_num = duration // interval_sec
    port = str(get_unique_port())
    enable = enable_azure

    max_rps = 100
    mps_infer = 30
    mps_train = 70


# MARK: Workload
## =========================================================== ##

def uniform(rps, client_model_list, infer_only=True, rps_fn=None,
            train_model:str = UniformConfig.train_model, 
            train_epoch:int = int(UniformConfig.duration / UniformConfig.train_epoch_time + 5), 
            train_batch_size:int = UniformConfig.train_batch_size):
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
           train_epoch:int = int(SkewedConfig.duration / SkewedConfig.train_epoch_time + 5), 
           train_batch_size:int = SkewedConfig.train_batch_size):
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

def azure(rps, client_model_list, infer_only=True, rps_fn=None,
          train_model:str = AzureConfig.train_model, 
          train_epoch:int = int(AzureConfig.duration / 5.5 + 5), 
          train_batch_size:int = AzureConfig.train_batch_size):
    workload = HyperWorkload(concurrency=2048,
                             warmup=5,
                             wait_warmup_done_sec=5,
                             wait_train_setup_sec=40,
                             wait_stable_before_start_profiling_sec=10)
    InferModel.reset_model_cnt()
    if not infer_only:
        workload.set_train_workload(
            train_workload=TrainWorkload(train_model, train_epoch, train_batch_size))
    workload.set_infer_workloads(AzureInferWorkload(
        AzureInferWorkload.TRACE_D01,
        model_list=client_model_list,
        max_request_sec=rps, 
        interval_sec=AzureConfig.interval_sec, 
        period_num=AzureConfig.period_num, 
        func_num=AzureConfig.num_model * 3, # 3 is a suitable number
        sort_trace_by='var_v2'
    ))
    return workload

def _run(system: System, workload: HyperWorkload, server_model_config: str, unit: str, tag: str):
    try:
        system.launch(unit, tag, time_stamp=use_time_stamp,
                    infer_model_config=server_model_config, fake_launch=False)
        workload.launch_workload(system, fake_launch=False)
        system.stop()
    except Exception as e:
        print(f"Failed to run {unit} {tag}: {e}")
        if retry_if_fail:
            print(f"\n\x1b[33;1m### Retry [{unit} {tag}] ###\x1b[0m")
            system.launch(unit, f'{tag}-retry', time_stamp=use_time_stamp,
                    infer_model_config=server_model_config)
            workload.launch_workload(system)
            system.stop()
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

## =========================================================== ##

## MARK: COLSYS
if run_colsys:
    system_config = {
        'mode' : System.ServerMode.ColocateL1,
        'use_sta' : True, 
        'mps' : True, 
        'use_xsched' : True, 
        'has_warmup' : True,
        'ondemand_adjust' : True,
        'cuda_memory_pool_gb' : "13.5",
        'train_memory_over_predict_mb' : 1500,
        'infer_model_max_idle_ms' : 5000,
        'cold_cache_ratio': 0.5, 
        'cold_cache_min_capability_nbytes': 1 * 1024 * 1024 * 1024,
        'cold_cache_max_capability_nbytes': int(1.5 * 1024 * 1024 * 1024),
    }

    if UniformConfig.enable and UniformConfig.high_load.enable:
        with mps_thread_percent(UniformConfig.high_load.mps_infer):
            client_model_list, server_model_config = InferModel.get_multi_model(
                UniformConfig.model_list, UniformConfig.num_model, 1)
            workload = uniform(rps=UniformConfig.high_load.rps, 
                               client_model_list=client_model_list, infer_only=False)
            system = System(train_mps_thread_percent=UniformConfig.high_load.mps_train,
                            port=UniformConfig.port,
                            **system_config)
            run(system, workload, server_model_config, "overall-uniform", "colsys-high")

    if UniformConfig.enable and UniformConfig.low_load.enable:
        with mps_thread_percent(UniformConfig.high_load.mps_infer):
            client_model_list, server_model_config = InferModel.get_multi_model(
                UniformConfig.model_list, UniformConfig.num_model, 1)
            workload = uniform(rps=UniformConfig.low_load.rps, 
                               client_model_list=client_model_list, infer_only=False)
            system = System(train_mps_thread_percent=UniformConfig.low_load.mps_train,
                            port=UniformConfig.port,
                            **system_config)
            run(system, workload, server_model_config, "overall-uniform", "colsys-low")

    if UniformConfig.enable and UniformConfig.hybrid_load.enable:
        with mps_thread_percent(UniformConfig.hybrid_load.mps_infer):
            client_model_list, server_model_config = InferModel.get_multi_model(
                UniformConfig.model_list, UniformConfig.num_model, 1)
            workload = uniform(rps=UniformConfig.hybrid_load.high_rps, 
                               client_model_list=client_model_list, infer_only=False)
            system = System(train_mps_thread_percent=UniformConfig.hybrid_load.mps_train,
                            port=UniformConfig.port,
                            **system_config)
            run(system, workload, server_model_config, "overall-uniform", "colsys-hybrid")


    if SkewedConfig.enable and SkewedConfig.high_load.enable:
        with mps_thread_percent(SkewedConfig.high_load.mps_infer):
            client_model_list, server_model_config = InferModel.get_multi_model(
                SkewedConfig.model_list, SkewedConfig.num_model, 1)
            workload = skewed(rps=SkewedConfig.high_load.rps, 
                              client_model_list=client_model_list, infer_only=False)
            system = System(train_mps_thread_percent=SkewedConfig.high_load.mps_train,
                            port=SkewedConfig.port,
                            **system_config)
            run(system, workload, server_model_config, "overall-skewed", "colsys-high")

    if SkewedConfig.enable and SkewedConfig.low_load.enable:
        with mps_thread_percent(SkewedConfig.low_load.mps_infer):
            client_model_list, server_model_config = InferModel.get_multi_model(
                SkewedConfig.model_list, SkewedConfig.num_model, 1)
            workload = skewed(rps=SkewedConfig.low_load.rps, 
                              client_model_list=client_model_list, infer_only=False)
            system = System(train_mps_thread_percent=SkewedConfig.low_load.mps_train,
                            port=SkewedConfig.port,
                            **system_config)
            run(system, workload, server_model_config, "overall-skewed", "colsys-low")

    if AzureConfig.enable:
        with mps_thread_percent(AzureConfig.mps_infer):
            client_model_list, server_model_config = InferModel.get_multi_model(
                AzureConfig.model_list, AzureConfig.num_model, 1)
            workload = azure(rps=AzureConfig.max_rps, 
                             client_model_list=client_model_list, infer_only=False)
            system = System(train_mps_thread_percent=AzureConfig.mps_train,
                            port=AzureConfig.port,
                            **system_config)
            run(system, workload, server_model_config, "overall-azure", "colsys")

    if enable_azure_profile_memory:
        with mps_thread_percent(AzureConfig.mps_infer):
            client_model_list, server_model_config = InferModel.get_multi_model(
                AzureConfig.model_list, AzureConfig.num_model, 1)
            workload = azure(rps=AzureConfig.max_rps, 
                             client_model_list=client_model_list, infer_only=False)
            system = System(train_mps_thread_percent=AzureConfig.mps_train,
                            port=AzureConfig.port,
                            profiler_acquire_resource_lock=True,
                            **system_config)
            run(system, workload, server_model_config, "overall-azure-memory-profile", "colsys")

    # only used for profiling memory
    if HybridLoad.enable:
        def rps_fn(i, rps):
            return HybridLoad.high_rps if i % 2 == 0 else HybridLoad.low_rps
        with mps_thread_percent(HybridLoad.mps_infer):
            client_model_list, server_model_config = InferModel.get_multi_model(
                UniformConfig.model_list, UniformConfig.num_model, 1)
            workload = uniform(rps=HybridLoad.high_rps, rps_fn=rps_fn,
                               client_model_list=client_model_list, infer_only=False)
            system = System(train_mps_thread_percent=HybridLoad.mps_train,
                            port=UniformConfig.port,
                            profiler_acquire_resource_lock=True,
                            **system_config)
            run(system, workload, server_model_config, "overall-hybrid", "colsys-hybrid")


## MARK: Task Switch
if run_task_switch:
    system_config = {
        'mode': System.ServerMode.TaskSwitchL1,
        'use_sta': True,
        'mps': False,
        'use_xsched': False,
        'has_warmup' : True,
        'cuda_memory_pool_gb': '13',
        'train_memory_over_predict_mb': 1500,
    }

    if UniformConfig.enable and UniformConfig.high_load.enable:
        client_model_list, server_model_config = InferModel.get_multi_model(
            UniformConfig.model_list, UniformConfig.num_model, 1)
        workload = uniform(rps=UniformConfig.high_load.rps, 
                           client_model_list=client_model_list, infer_only=False)
        system = System(port=UniformConfig.port, **system_config)
        run(system, workload, server_model_config, "overall-uniform", "task-switch-high")

    if UniformConfig.enable and UniformConfig.low_load.enable:
        client_model_list, server_model_config = InferModel.get_multi_model(
            UniformConfig.model_list, UniformConfig.num_model, 1)
        workload = uniform(rps=UniformConfig.low_load.rps, 
                           client_model_list=client_model_list, infer_only=False)
        system = System(port=UniformConfig.port, **system_config)
        run(system, workload, server_model_config, "overall-uniform", "task-switch-low")

    if SkewedConfig.enable and SkewedConfig.high_load.enable:
        client_model_list, server_model_config = InferModel.get_multi_model(
            SkewedConfig.model_list, SkewedConfig.num_model, 1)
        workload = skewed(rps=SkewedConfig.high_load.rps, 
                          client_model_list=client_model_list, infer_only=False)
        system = System(port=SkewedConfig.port, **system_config)
        run(system, workload, server_model_config, "overall-skewed", "task-switch-high")

    if SkewedConfig.enable and SkewedConfig.low_load.enable:
        client_model_list, server_model_config = InferModel.get_multi_model(
            SkewedConfig.model_list, SkewedConfig.num_model, 1)
        workload = skewed(rps=SkewedConfig.low_load.rps, 
                          client_model_list=client_model_list, infer_only=False)
        system = System(port=SkewedConfig.port, **system_config)
        run(system, workload, server_model_config, "overall-skewed", "task-switch-low")

    if AzureConfig.enable:
        client_model_list, server_model_config = InferModel.get_multi_model(
            AzureConfig.model_list, AzureConfig.num_model, 1)
        workload = azure(rps=AzureConfig.max_rps, 
                         client_model_list=client_model_list, infer_only=False)
        system = System(port=AzureConfig.port, **system_config)
        run(system, workload, server_model_config, "overall-azure", "task-switch")


## MARK: UM+MPS
if run_um_mps:
    system_config = {
        'mode' : System.ServerMode.Normal,
        'use_sta': False,
        'mps': True,
        'use_xsched': False,
        'has_warmup': True,
    }

    if UniformConfig.enable and UniformConfig.high_load.enable:
        with um_mps(UniformConfig.high_load.mps_infer):
            client_model_list, server_model_config = InferModel.get_multi_model(
                UniformConfig.model_list, UniformConfig.num_model, 1)
            workload = uniform(UniformConfig.high_load.rps, client_model_list, infer_only=False)
            system = System(train_mps_thread_percent=UniformConfig.high_load.mps_train,
                            port=UniformConfig.port, **system_config)
            run(system, workload, server_model_config, "overall-uniform", "um-mps-high")

    if UniformConfig.enable and UniformConfig.low_load.enable:
        with um_mps(UniformConfig.low_load.mps_infer):
            client_model_list, server_model_config = InferModel.get_multi_model(
                UniformConfig.model_list, UniformConfig.num_model, 1)
            workload = uniform(UniformConfig.low_load.rps, client_model_list, infer_only=False)
            system = System(train_mps_thread_percent=UniformConfig.low_load.mps_train,
                            port=UniformConfig.port, **system_config)
            run(system, workload, server_model_config, "overall-uniform", "um-mps-low")

    if SkewedConfig.enable and SkewedConfig.high_load.enable:
        with um_mps(SkewedConfig.high_load.mps_infer):
            client_model_list, server_model_config = InferModel.get_multi_model(
                SkewedConfig.model_list, SkewedConfig.num_model, 1)
            workload = skewed(SkewedConfig.high_load.rps, client_model_list, infer_only=False)
            system = System(train_mps_thread_percent=SkewedConfig.high_load.mps_train,
                            port=SkewedConfig.port, **system_config)
            run(system, workload, server_model_config, "overall-skewed", "um-mps-high")

    if SkewedConfig.enable and SkewedConfig.low_load.enable:
        with um_mps(SkewedConfig.low_load.mps_infer):
            client_model_list, server_model_config = InferModel.get_multi_model(
                SkewedConfig.model_list, SkewedConfig.num_model, 1)
            workload = skewed(SkewedConfig.low_load.rps, client_model_list, infer_only=False)
            system = System(train_mps_thread_percent=SkewedConfig.low_load.mps_train,
                            port=SkewedConfig.port, **system_config)
            run(system, workload, server_model_config, "overall-skewed", "um-mps-low")

    if AzureConfig.enable:
        with um_mps(AzureConfig.mps_infer):
            client_model_list, server_model_config = InferModel.get_multi_model(
                AzureConfig.model_list, AzureConfig.num_model, 1)
            workload = azure(rps=AzureConfig.max_rps, 
                             client_model_list=client_model_list, infer_only=False)
            system = System(train_mps_thread_percent=AzureConfig.mps_train,
                            port=AzureConfig.port, **system_config)
            run(system, workload, server_model_config, "overall-azure", "um-mps")


## MARK: Static Partition
for tag, item in {
    'F': ({
        'mode' : System.ServerMode.Normal,
        'use_sta': True,
        'mps': True,
        'use_xsched': False,
        'has_warmup': True,
        'max_warm_cache_nbytes': int(5.5 * 1024 ** 3),
        'cuda_memory_pool_gb': '7.5',
        'use_sta_train': False
    }, {'train_batch_size': 32}),
    'I': ({
        'mode' : System.ServerMode.Normal,
        'use_sta': True,
        'mps': True,
        'use_xsched': False,
        'has_warmup': True,
        'max_warm_cache_nbytes': int(9 * 1024 ** 3),
        'cuda_memory_pool_gb': '10.5',
        'use_sta_train': False
    }, {'train_batch_size': 8}), 
}.items():
    if not run_static_partition:
        break
    if tag == 'F' and not run_static_partition_F:
        continue
    if tag == 'I' and not run_static_partition_I:
        continue
    system_config, workload_config = item
    if UniformConfig.enable and UniformConfig.high_load.enable:
        with mps_thread_percent(UniformConfig.high_load.mps_infer):
            client_model_list, server_model_config = InferModel.get_multi_model(
                UniformConfig.model_list, UniformConfig.num_model, 1)
            workload = uniform(rps=UniformConfig.high_load.rps, 
                               client_model_list=client_model_list, infer_only=False,
                               train_batch_size=workload_config['train_batch_size'])
            system = System(train_mps_thread_percent=UniformConfig.high_load.mps_train,
                            port=UniformConfig.port, **system_config)
            run(system, workload, server_model_config, "overall-uniform", f"static-partition-high-{tag}")


    if UniformConfig.enable and UniformConfig.low_load.enable:
        with mps_thread_percent(UniformConfig.low_load.mps_infer):
            client_model_list, server_model_config = InferModel.get_multi_model(
                UniformConfig.model_list, UniformConfig.num_model, 1)
            workload = uniform(rps=UniformConfig.low_load.rps, 
                               client_model_list=client_model_list, infer_only=False,
                               train_batch_size=workload_config['train_batch_size'])
            system = System(train_mps_thread_percent=UniformConfig.low_load.mps_train,
                            port=UniformConfig.port, **system_config)
            run(system, workload, server_model_config, "overall-uniform", f"static-partition-low-{tag}")

    if SkewedConfig.enable and SkewedConfig.high_load.enable:
        with mps_thread_percent(SkewedConfig.high_load.mps_infer):
            client_model_list, server_model_config = InferModel.get_multi_model(
                SkewedConfig.model_list, SkewedConfig.num_model, 1)
            workload = skewed(rps=SkewedConfig.high_load.rps, 
                              client_model_list=client_model_list, infer_only=False,
                              train_batch_size=workload_config['train_batch_size'])
            system = System(train_mps_thread_percent=SkewedConfig.high_load.mps_train,
                            port=SkewedConfig.port, **system_config)
            run(system, workload, server_model_config, "overall-skewed", f"static-partition-high-{tag}")

    if SkewedConfig.enable and SkewedConfig.low_load.enable:
        with mps_thread_percent(SkewedConfig.low_load.mps_infer):
            client_model_list, server_model_config = InferModel.get_multi_model(
                SkewedConfig.model_list, SkewedConfig.num_model, 1)
            workload = skewed(rps=SkewedConfig.low_load.rps, 
                              client_model_list=client_model_list, infer_only=False,
                              train_batch_size=workload_config['train_batch_size'])
            system = System(train_mps_thread_percent=SkewedConfig.low_load.mps_train,
                            port=SkewedConfig.port, **system_config)
            run(system, workload, server_model_config, "overall-skewed", f"static-partition-low-{tag}")

    if AzureConfig.enable:
        with mps_thread_percent(AzureConfig.mps_infer):
            client_model_list, server_model_config = InferModel.get_multi_model(
                AzureConfig.model_list, AzureConfig.num_model, 1)
            workload = azure(rps=AzureConfig.max_rps, 
                             client_model_list=client_model_list, infer_only=False,
                             train_batch_size=workload_config['train_batch_size'])
            system = System(train_mps_thread_percent=AzureConfig.mps_train,
                            port=AzureConfig.port, **system_config)
            run(system, workload, server_model_config, "overall-azure", f"static-partition-{tag}")


## MARK: Infer Only
if run_infer_only:
    system_config = {
        'mode' : System.ServerMode.Normal,
        'use_sta' : False,
        'mps': True,
        'use_xsched': False,
        'has_warmup': True,
    }

    mps_tag = "" if not infer_only_without_mps else "-no-mps"

    if UniformConfig.enable and UniformConfig.high_load.enable:
        with mps_thread_percent(UniformConfig.high_load.mps_infer, skip=infer_only_without_mps):
            client_model_list, server_model_config = InferModel.get_multi_model(
                UniformConfig.model_list, UniformConfig.num_model, 1)
            workload = uniform(UniformConfig.high_load.rps, client_model_list, infer_only=True)
            system = System(port=UniformConfig.port, **system_config)
            run(system, workload, server_model_config, "overall-uniform", f"infer-only-high{mps_tag}")

    if UniformConfig.enable and UniformConfig.low_load.enable:
        # with mps_thread_percent(UniformConfig.)
        with mps_thread_percent(UniformConfig.low_load.mps_infer, skip=infer_only_without_mps):
            client_model_list, server_model_config = InferModel.get_multi_model(
                UniformConfig.model_list, UniformConfig.num_model, 1)
            workload = uniform(UniformConfig.low_load.rps, client_model_list, infer_only=True)
            system = System(port=UniformConfig.port, **system_config)
            run(system, workload, server_model_config, "overall-uniform", f"infer-only-low{mps_tag}")

    if SkewedConfig.enable and SkewedConfig.high_load.enable:
        with mps_thread_percent(SkewedConfig.high_load.mps_infer, skip=infer_only_without_mps):
            client_model_list, server_model_config = InferModel.get_multi_model(
                SkewedConfig.model_list, SkewedConfig.num_model, 1)
            workload = skewed(SkewedConfig.high_load.rps, client_model_list, infer_only=True)
            system = System(port=SkewedConfig.port, **system_config)
            run(system, workload, server_model_config, "overall-skewed", f"infer-only-high{mps_tag}")

    if SkewedConfig.enable and SkewedConfig.low_load.enable:
        with mps_thread_percent(SkewedConfig.low_load.mps_infer, skip=infer_only_without_mps):
            client_model_list, server_model_config = InferModel.get_multi_model(
                SkewedConfig.model_list, SkewedConfig.num_model, 1)
            workload = skewed(SkewedConfig.low_load.rps, client_model_list, infer_only=True)
            system = System(port=SkewedConfig.port, **system_config)
            run(system, workload, server_model_config, "overall-skewed", f"infer-only-low{mps_tag}")

    if AzureConfig.enable:
        with mps_thread_percent(AzureConfig.mps_infer, skip=infer_only_without_mps):
            client_model_list, server_model_config = InferModel.get_multi_model(
                AzureConfig.model_list, AzureConfig.num_model, 1)
            workload = azure(rps=AzureConfig.max_rps, 
                             client_model_list=client_model_list, infer_only=True)
            system = System(port=AzureConfig.port, **system_config)
            run(system, workload, server_model_config, "overall-azure", f"infer-only{mps_tag}")


