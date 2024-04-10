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
run_infer_only = False

infer_only_without_mps = False

# args parser
parser = argparse.ArgumentParser()
parser.add_argument('--colsys', action='store_true')
parser.add_argument('--um-mps', action='store_true')
parser.add_argument('--task-switch', action='store_true')
parser.add_argument('--static-partition', action='store_true')
parser.add_argument('--infer-only', action='store_true')
parser.add_argument('--all', action='store_true')
args = parser.parse_args()

if args.colsys or args.all:
    run_colsys = True
if args.um_mps or args.all:
    run_um_mps = True
if args.task_switch or args.all:
    run_task_switch = True
if args.static_partition or args.all:
    run_static_partition = True
if args.infer_only or args.all:
    run_infer_only = True

## MARK: Configurations
## =========================================================== ##

@dataclass
class HighLoad:
    rps: int = 50
    mps_infer: int = 40
    mps_train: int = 60
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
    

class UniformConfig:
    model_list = [InferModel.InceptionV3, InferModel.ResNet152, 
                  InferModel.DenseNet161, InferModel.DistilBertBase]
    num_model = 60
    interval_sec = 20
    duration = 120
    port = str(18100 + (os.getpid() % 10) * 10)
    enable = True

    low_load = LowLoad(enable=False)
    high_load = HighLoad(enable=True)
    hybrid_load = HybridLoad(enable=False)

# MARK: Workload
## =========================================================== ##

def uniform(rps, client_model_list, infer_only=True, rps_fn=None,
            train_model:str ='resnet', 
            train_epoch:int = int(UniformConfig.duration / 3 + 5), 
            train_batch_size:int = 130):
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


def _run(system: System, workload: HyperWorkload, server_model_config: str, unit: str, tag: str):
    try:
        system.launch(unit, tag, time_stamp=use_time_stamp,
                    infer_model_config=server_model_config)
        workload.launch_workload(system)
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
    if skip_fail:
        try:
            _run(system, workload, server_model_config, unit, tag)
        except Exception as e:
            print(f"\n\x1b[33;1m### Skip {unit} {tag}: {e} ###\x1b[0m")
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


## MARK: Static Partition
if run_static_partition:
    system_config = {
        'mode' : System.ServerMode.Normal,
        'use_sta': True,
        'mps': True,
        'use_xsched': False,
        'has_warmup': True,
        'max_warm_cache_nbytes': int(8.5 * 1024 ** 3),
        'cuda_memory_pool_gb': '10',
        'use_sta_train': False
    }
    train_batch_size = 20

    if UniformConfig.enable and UniformConfig.high_load.enable:
        with mps_thread_percent(UniformConfig.high_load.mps_infer):
            client_model_list, server_model_config = InferModel.get_multi_model(
                UniformConfig.model_list, UniformConfig.num_model, 1)
            workload = uniform(rps=UniformConfig.high_load.rps, 
                               client_model_list=client_model_list, infer_only=False,
                               train_batch_size=train_batch_size)
            system = System(train_mps_thread_percent=UniformConfig.high_load.mps_train,
                            port=UniformConfig.port, **system_config)
            run(system, workload, server_model_config, "overall-uniform", "static-partition-high")


    if UniformConfig.enable and UniformConfig.low_load.enable:
        with mps_thread_percent(UniformConfig.low_load.mps_infer):
            client_model_list, server_model_config = InferModel.get_multi_model(
                UniformConfig.model_list, UniformConfig.num_model, 1)
            workload = uniform(rps=UniformConfig.low_load.rps, 
                               client_model_list=client_model_list, infer_only=False,
                               train_batch_size=train_batch_size)
            system = System(train_mps_thread_percent=UniformConfig.low_load.mps_train,
                            port=UniformConfig.port, **system_config)
            run(system, workload, server_model_config, "overall-uniform", "static-partition-low")


## MARK: Infer Only
if run_infer_only:
    system_config = {
        'mode' : System.ServerMode.Normal,
        'use_sta' : False,
        'mps': True,
        'use_xsched': False,
        'has_warmup': True,
    }

    if UniformConfig.enable and UniformConfig.high_load.enable:
        with mps_thread_percent(UniformConfig.high_load.mps_infer, skip=infer_only_without_mps):
            client_model_list, server_model_config = InferModel.get_multi_model(
                UniformConfig.model_list, UniformConfig.num_model, 1)
            workload = uniform(UniformConfig.high_load.rps, client_model_list, infer_only=True)
            system = System(port=UniformConfig.port, **system_config)
            run(system, workload, server_model_config, "overall-uniform", "infer-only-high")

    if UniformConfig.enable and UniformConfig.low_load.enable:
        # with mps_thread_percent(UniformConfig.)
        with mps_thread_percent(UniformConfig.low_load.mps_infer, skip=infer_only_without_mps):
            client_model_list, server_model_config = InferModel.get_multi_model(
                UniformConfig.model_list, UniformConfig.num_model, 1)
            workload = uniform(UniformConfig.low_load.rps, client_model_list, infer_only=True)
            system = System(port=UniformConfig.port, **system_config)
            run(system, workload, server_model_config, "overall-uniform", "infer-only-low")
            