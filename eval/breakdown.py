import os
import argparse
from runner import *
set_global_seed(42)

from dataclasses import dataclass
import run_comm

run_comm.use_time_stamp = True
run_comm.retry_if_fail = True
run_comm.skip_fail = True

run_comm.UniformConfig.duration = 300
run_comm.SkewedConfig.duration = 300

# run_comm.fake_launch = True



run_colsys  = False
run_strawman = False

enable_uniform = False
enable_skewed = False
enable_azure = False
enable_uniform_v2 = False


run_comm.UniformConfig_v2.model_list = [InferModel.ResNet152]
run_comm.UniformConfig_v2.num_model = 32
uniform_v2_wkld_type = 'NormalC'

# args parser
parser = argparse.ArgumentParser()
parser.add_argument('--colsys', action='store_true')
parser.add_argument('--strawman', action='store_true')
parser.add_argument('--uniform', action='store_true')
parser.add_argument('--uniform-v2', action='store_true')
parser.add_argument('--skewed', action='store_true')
parser.add_argument('--azure', action='store_true')
parser.add_argument('--all-sys', action='store_true')
parser.add_argument('--all-workload', action='store_true')
parser.add_argument('--multi-gpu', action='store_true')
parser.add_argument('--retry-limit', type=int, default=0)
parser.add_argument('--parse-result', action='store_true')
args = parser.parse_args()

if args.colsys or args.all_sys:
    run_colsys = True
if args.strawman or args.all_sys:
    run_strawman = True

if args.uniform or args.all_workload:
    enable_uniform = True
if args.skewed or args.all_workload:
    enable_skewed = True
if args.azure or args.all_workload:
    enable_azure = True
if args.uniform_v2 or args.all_workload:
    enable_uniform_v2 = True

if args.retry_limit > 0:
    run_comm.retry_if_fail = True
    run_comm.retry_limit = args.retry_limit
    run_comm.skip_fail = True

if args.multi_gpu:
    run_comm.UniformConfig_v2.train_model += "_ddp"
    run_comm.SkewedConfig_v2.train_model += "_ddp"

if args.parse_result:
    LogParser._enable = True


# MARK: Trace Config
class UniformConfig:
    train_model = 'swin_b' if not args.multi_gpu else 'swin_b_ddp'
    train_batch_size = 72
    train_global_batch_size = 500 # not used, hard code for global batch size and dataset size
    train_dataset_size = 1000 
    train_epoch_time = 5.5 # used for predict number epoch

    model_list = [InferModel.ResNet152]
    num_model = 32
    interval_sec = run_comm.UniformConfig.interval_sec
    duration = run_comm.UniformConfig.duration
    port = str(run_comm.get_unique_port())
    enable = enable_uniform

    low_load = run_comm.LowLoad(enable=False)
    high_load = run_comm.HighLoad(enable=True)
    hybrid_load = run_comm.HybridLoad(enable=False)


class SkewedConfig:
    train_model = 'swin_b' if not args.multi_gpu else 'swin_b_ddp'
    train_batch_size = 72
    train_global_batch_size = 500 # not used, hard code for global batch size and dataset size
    train_dataset_size = 1000 
    train_epoch_time = 5.5 # used for predict number epoch

    model_list = [InferModel.ResNet152]
    num_model = 32
    interval_sec = run_comm.SkewedConfig.interval_sec
    duration = run_comm.SkewedConfig.duration
    zipf_aplha = 1.05 # large alpha -> more skewed
    port = str(run_comm.get_unique_port())
    enable = enable_skewed

    low_load = run_comm.LowLoad(enable=False) # will not used
    high_load = run_comm.HighLoad(enable=True)
    hybrid_load = run_comm.HybridLoad(enable=False)


class AzureConfig:
    train_model = 'swin_b' if not args.multi_gpu else 'swin_b_ddp'
    train_batch_size = 72 if not args.multi_gpu else 66

    # model_list = [InferModel.DenseNet161, InferModel.EfficientNetV2_s, 
    #               InferModel.EfficientViT_b2, InferModel.DistilBertBase, 
    #               InferModel.ResNet152, InferModel.DistilGPT2] 
    model_list = [InferModel.ResNet152]
    # num_model = 64
    num_model = runner.scale_up_by_num_gpu(32) # or 30
    interval_sec = 5
    duration = 300 
    period_num = duration // interval_sec
    port = str(run_comm.get_unique_port())
    enable = enable_azure

    max_rps = runner.scale_up_by_num_gpu(150)
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
    workload.set_infer_workloads(MicrobenchmarkInferWorkload_v1(
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
    workload.set_infer_workloads(MicrobenchmarkInferWorkload_v1(
        model_list=client_model_list,
        interval_sec=SkewedConfig.interval_sec, fix_request_sec=rps, rps_fn=rps_fn,
        zipf_alpha=SkewedConfig.zipf_aplha,
        duration=SkewedConfig.duration + workload.infer_extra_infer_sec,
    ))
    return workload


def azure(rps, client_model_list, infer_only=True, rps_fn=None,
          train_model:str = AzureConfig.train_model, 
          train_epoch:int = int(AzureConfig.duration + 5), 
          train_batch_size:int = AzureConfig.train_batch_size):
    print(f'azure rps {rps}')
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


## =========================================================== ##

## MARK: COLSYS
if run_colsys:
    system_config = {
        'mode' : System.ServerMode.ColocateL1,
        'use_sta' : True, 
        'mps' : True, 
        'skip_set_mps_thread_percent': True,
        'use_xsched' : True, 
        'has_warmup' : True,
        'ondemand_adjust' : True,
        'cuda_memory_pool_gb' : "13" if not args.multi_gpu else "12.5",
        'train_memory_over_predict_mb' : 1500,
        'infer_model_max_idle_ms' : 5000,
        'cold_cache_ratio': 0.5, 
        'cold_cache_min_capacity_nbytes': int(0 * 1024 * 1024 * 1024),
        'cold_cache_max_capacity_nbytes': int(2 * 1024 * 1024 * 1024),
        'dynamic_sm_partition': True,
    }

    if UniformConfig.enable and UniformConfig.high_load.enable:
        with mps_thread_percent(None):
            client_model_list, server_model_config = InferModel.get_multi_model(
                UniformConfig.model_list, UniformConfig.num_model, 1)
            workload = uniform(rps=UniformConfig.high_load.rps, 
                               client_model_list=client_model_list, 
                               infer_only=False)
            system = System(port=UniformConfig.port,
                            dump_adjust_info=True,
                            **system_config)
            run_comm.run(system, workload, server_model_config, 
                         "breakdown-uniform", "colsys-high")

    if SkewedConfig.enable and SkewedConfig.high_load.enable:
        with mps_thread_percent(SkewedConfig.high_load.mps_infer):
            client_model_list, server_model_config = InferModel.get_multi_model(
                SkewedConfig.model_list, SkewedConfig.num_model, 1)
            workload = skewed(rps=SkewedConfig.high_load.rps, 
                              client_model_list=client_model_list, infer_only=False)
            system = System(train_mps_thread_percent=SkewedConfig.high_load.mps_train,
                            port=SkewedConfig.port,
                            dump_adjust_info=True,
                            **system_config)
            run_comm.run(system, workload, server_model_config, "breakdown-skewed", "colsys-high")

    if SkewedConfig.enable and SkewedConfig.low_load.enable:
        with mps_thread_percent(SkewedConfig.low_load.mps_infer):
            client_model_list, server_model_config = InferModel.get_multi_model(
                SkewedConfig.model_list, SkewedConfig.num_model, 1)
            workload = skewed(rps=SkewedConfig.low_load.rps, 
                              client_model_list=client_model_list, infer_only=False)
            system = System(train_mps_thread_percent=SkewedConfig.low_load.mps_train,
                            port=SkewedConfig.port,
                            dump_adjust_info=True,
                            **system_config)
            run_comm.run(system, workload, server_model_config, "breakdown-skewed", "colsys-low")

    if enable_uniform_v2:
        with mps_thread_percent(None):
            client_model_list, server_model_config = InferModel.get_multi_model(
                run_comm.UniformConfig_v2.model_list, run_comm.UniformConfig_v2.num_model, 1)
            workload = run_comm.uniform_v2(uniform_v2_wkld_type, client_model_list, infer_only=False)
            system = System(port=run_comm.UniformConfig_v2.port,
                            dump_adjust_info=True,
                            **system_config)
            run_comm.run(system, workload, server_model_config, 
                         f"breakdown-uniform-v2-{runner.get_num_gpu()}gpu", 
                         f"colsys-{uniform_v2_wkld_type}")
            
    if AzureConfig.enable:
        with mps_thread_percent(None):
            client_model_list, server_model_config = InferModel.get_multi_model(
                AzureConfig.model_list, AzureConfig.num_model, 1)
            workload = azure(rps=AzureConfig.max_rps,
                             client_model_list=client_model_list, 
                             infer_only=False)
            system = System(port=AzureConfig.port,
                            dump_adjust_info=True,
                            **system_config)
            run_comm.run(system, workload, server_model_config,
                         f"breakdown-azure-{runner.get_num_gpu()}gpu", 
                         "colsys")


## MARK: Strawman
if run_strawman:
    system_config = {
        'mode' : System.ServerMode.ColocateL2,
        'use_sta' : False, 
        'mps' : True, 
        'skip_set_mps_thread_percent': True,
        'use_xsched' : True, 
        'has_warmup' : True,
        'ondemand_adjust' : True,
        'pipeline_load': False,
        'train_memory_over_predict_mb' : 1000,
        # 'cuda_memory_pool_gb' : "13.5",
        # 'infer_model_max_idle_ms' : 5000,
        # 'cold_cache_ratio': 0.5, 
        # 'cold_cache_min_capacity_nbytes': 1 * 1024 * 1024 * 1024,
        # 'cold_cache_max_capacity_nbytes': int(1.5 * 1024 * 1024 * 1024),
        'dynamic_sm_partition': True,
    }

    if UniformConfig.enable and UniformConfig.high_load.enable:
        client_model_list, server_model_config = InferModel.get_multi_model(
            UniformConfig.model_list, UniformConfig.num_model, 1)
        workload = uniform(rps=UniformConfig.high_load.rps, 
                           client_model_list=client_model_list, infer_only=False)
        system = System(port=UniformConfig.port,
                        dump_adjust_info=True,
                        **system_config)
        run_comm.run(system, workload, server_model_config, 
                     "breakdown-uniform", "strawman-high")

    if SkewedConfig.enable and SkewedConfig.high_load.enable:
        client_model_list, server_model_config = InferModel.get_multi_model(
            SkewedConfig.model_list, SkewedConfig.num_model, 1)
        workload = skewed(rps=SkewedConfig.high_load.rps, 
                          client_model_list=client_model_list, infer_only=False)
        system = System(train_mps_thread_percent=SkewedConfig.high_load.mps_train,
                        port=SkewedConfig.port,
                        dump_adjust_info=True,
                        **system_config)
        run_comm.run(system, workload, server_model_config, 
                     "breakdown-skewed", "strawman-high")
        
    if enable_uniform_v2:
        with mps_thread_percent(None):
            client_model_list, server_model_config = InferModel.get_multi_model(
                run_comm.UniformConfig_v2.model_list, run_comm.UniformConfig_v2.num_model, 1)
            workload = run_comm.uniform_v2(uniform_v2_wkld_type, 
                                           client_model_list, infer_only=False)
            system = System(port=run_comm.UniformConfig_v2.port,
                            dump_adjust_info=True,
                            **system_config)
            run_comm.run(system, workload, server_model_config, 
                         f"breakdown-uniform-v2-{runner.get_num_gpu()}gpu", 
                         f"strawman-{uniform_v2_wkld_type}")

    if AzureConfig.enable:
        with mps_thread_percent(None):
            client_model_list, server_model_config = InferModel.get_multi_model(
                AzureConfig.model_list, AzureConfig.num_model, 1)
            workload = azure(rps=AzureConfig.max_rps,
                             client_model_list=client_model_list, 
                             infer_only=False)
            system = System(port=AzureConfig.port,
                            dump_adjust_info=True,
                            **system_config)
            run_comm.run(system, workload, server_model_config, 
                         f"breakdown-azure-{runner.get_num_gpu()}gpu", 
                         "strawman")
                        
        
# =========================================================
# Parse result
# =========================================================
if LogParser._enable:
    if args.multi_gpu:
        LogParser.parse(TestUnit.BREAKDOWN_MULTI_GPU)
    else:
        LogParser.parse(TestUnit.BREAKDOWN_SINGLE_GPU)