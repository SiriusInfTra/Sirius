import os
import argparse
from runner import *
from dataclasses import dataclass
import run_comm
set_global_seed(42)

run_comm.use_time_stamp = True
run_comm.retry_if_fail = True
run_comm.skip_fail = True
run_comm.retry_limit = 3

def get_unique_port():
    cuda_device_env = os.environ['CUDA_VISIBLE_DEVICES']
    # assert ',' not in cuda_device
    cuda_device = cuda_device_env.split(',')[0]
    try:
        cuda_device = int(cuda_device)
    except:
        cuda_device = GPU_UUIDs.index(cuda_device)
    port = 18100 if not runner.is_inside_docker() else 18200
    port += cuda_device
    return port


class AzureConfig:
    train_model = 'swin_b'
    train_batch_size = 72

    model_list = [InferModel.DenseNet161, InferModel.EfficientNetV2_s, 
                  InferModel.EfficientViT_b2, InferModel.DistilBertBase, 
                  InferModel.ResNet152, InferModel.DistilGPT2] 
    # num_model = 64
    num_model = 56
    interval_sec = 5
    duration = 300 
    period_num = duration // interval_sec
    port = str(get_unique_port())
    enable = True

    max_rps = 150
    mps_infer = 30
    mps_train = 70


def azure(rps, client_model_list, infer_only=False, rps_fn=None,
          train_model:str = AzureConfig.train_model, 
          train_epoch:int = int(AzureConfig.duration / 5.5 + 5), 
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

set_global_seed(42)

enable_eval_max_infer_idle_ms = False
enable_eval_cold_cache_cap = False

# args parser
parser = argparse.ArgumentParser()
parser.add_argument('--eval-max-infer-idle-ms', action='store_true')
parser.add_argument('--eval-cold-cache-cap', action='store_true')
parser.add_argument('--eval-all', action='store_true')
parser.add_argument('--retry-limit', type=int, default=0)
parser.add_argument('--parse-result', action='store_true')
args = parser.parse_args()

if args.eval_max_infer_idle_ms or args.eval_all:
    enable_eval_max_infer_idle_ms = True
if args.eval_cold_cache_cap or args.eval_all:
    enable_eval_cold_cache_cap = True

if args.retry_limit > 0:
    run_comm.retry_if_fail = True
    run_comm.retry_limit = args.retry_limit
    run_comm.skip_fail = True

if args.parse_result:
    LogParser._enable = True


max_infer_idle_ms_list = [
    500, 
    2500,
    5000, 
    7500, 
    10000,
]
if not enable_eval_max_infer_idle_ms:
    max_infer_idle_ms_list = []

cold_cache_cap_list = [
    (0, 0), 
    (0, 1), 
    (0, 2), 
    (0, 3),
    (0, 4),
    (0, 5),
]
if not enable_eval_cold_cache_cap:
    cold_cache_cap_list = []


# MARK: max_infer_idle_ms
for max_infer_idle_ms in max_infer_idle_ms_list:
    system_config = {
        'mode' : System.ServerMode.ColocateL1,
        'use_sta' : True, 
        'mps' : True, 
        'skip_set_mps_thread_percent': True,
        'use_xsched' : True,
        'has_warmup' : True,
        'ondemand_adjust' : True,
        'cuda_memory_pool_gb' : "13",
        'train_memory_over_predict_mb' : 1500,
        'infer_model_max_idle_ms' : max_infer_idle_ms,
        'cold_cache_ratio': 0.5, 
        'cold_cache_min_capacity_nbytes': 0,
        'cold_cache_max_capacity_nbytes': 0,
        'dynamic_sm_partition': True,
    }

    with mps_thread_percent(None):
        client_model_list, server_model_config = InferModel.get_multi_model(
            AzureConfig.model_list, AzureConfig.num_model, 1)
        workload = azure(rps=AzureConfig.max_rps, 
                            client_model_list=client_model_list)
        system = System(port=AzureConfig.port, **system_config)
        run_comm.run(system, workload, server_model_config, 
                     "ablation-infer-idle-time-uniform", f"{max_infer_idle_ms}ms")
        

# MARK: cold cache cap
for cold_cache_min_cap, cold_cache_max_cap in cold_cache_cap_list:
    system_config = {
        'mode' : System.ServerMode.ColocateL1,
        'use_sta' : True, 
        'mps' : True, 
        'skip_set_mps_thread_percent': True,
        'use_xsched' : True,
        'has_warmup' : True,
        'ondemand_adjust' : True,
        'cuda_memory_pool_gb' : "13",
        'train_memory_over_predict_mb' : 1500,
        'infer_model_max_idle_ms' : 500,
        'cold_cache_ratio': 0.5, 
        'cold_cache_min_capacity_nbytes': int(cold_cache_min_cap * 1024 * 1024 * 1024),
        'cold_cache_max_capacity_nbytes': int(cold_cache_max_cap * 1024 * 1024 * 1024),
        'dynamic_sm_partition': True,
    }
        
    with mps_thread_percent(None):
        client_model_list, server_model_config = InferModel.get_multi_model(
            AzureConfig.model_list, AzureConfig.num_model, 1)
        workload = azure(rps=AzureConfig.max_rps, 
                            client_model_list=client_model_list)
        system = System(port=AzureConfig.port, **system_config)
        run_comm.run(system, workload, server_model_config, 
                     "ablation-cold-cache-cap-uniform", 
                     f"{cold_cache_min_cap}GB-{cold_cache_max_cap}GB")


# =========================================================
# Parse result
# =========================================================
if LogParser._enable:
    LogParser.parse(TestUnit.ABLATION)