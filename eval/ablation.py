import os
import argparse
from runner import *
from dataclasses import dataclass
import run_comm

run_comm.use_time_stamp = True
run_comm.retry_if_fail = True
run_comm.skip_fail = True
run_comm.retry_limit = 3

run_comm.UniformConfig.duration = 300
run_comm.SkewedConfig.duration = 300


set_global_seed(42)

enable_eval_max_infer_idle_ms = False
enable_eval_cold_cache_cap = False

# args parser
parser = argparse.ArgumentParser()
parser.add_argument('--eval-max-infer-idle-ms', action='store_true')
parser.add_argument('--eval-cold-cache-cap', action='store_true')
parser.add_argument('--eval-all', action='store_true')
args = parser.parse_args()

if args.eval_max_infer_idle_ms or args.eval_all:
    enable_eval_max_infer_idle_ms = True
if args.eval_cold_cache_cap or args.eval_all:
    enable_eval_cold_cache_cap = True


max_infer_idle_ms_list = [
    # 500, 
    2500,
    # 5000, 
    10000,
    7500, 
]
if not enable_eval_max_infer_idle_ms:
    max_infer_idle_ms_list = []

cold_cache_cap_list = [
    (0, 0), 
    (0.5, 1), 
    (1, 2), 
    (1.5, 3)
]
if not enable_eval_cold_cache_cap:
    cold_cache_cap_list = []


# MARK: max_infer_idle_ms
for max_infer_idle_ms in max_infer_idle_ms_list:
    system_config = {
        'mode' : System.ServerMode.ColocateL1,
        'use_sta' : True, 
        'mps' : True, 
        'use_xsched' : True, 
        'has_warmup' : True,
        'ondemand_adjust' : True,
        'cuda_memory_pool_gb' : "13.5",
        'train_memory_over_predict_mb' : 1500,
        'infer_model_max_idle_ms' : max_infer_idle_ms,
        'cold_cache_ratio': 0.5, 
        # 'cold_cache_min_capability_nbytes': 1 * 1024 * 1024 * 1024,
        # 'cold_cache_max_capability_nbytes': int(1.5 * 1024 * 1024 * 1024),
        'cold_cache_min_capability_nbytes': 0,
        'cold_cache_max_capability_nbytes': 0,
    }

    with mps_thread_percent(run_comm.UniformConfig.high_load.mps_infer):
        client_model_list, server_model_config = InferModel.get_multi_model(
                run_comm.UniformConfig.model_list, run_comm.UniformConfig.num_model, 1)
        workload = run_comm.uniform(rps=run_comm.UniformConfig.high_load.rps, 
                                    client_model_list=client_model_list, infer_only=False,
                                    train_epoch=run_comm.get_train_epoch(run_comm.UniformConfig.train_epoch_time, run_comm.UniformConfig.duration))
        system = System(train_mps_thread_percent=run_comm.UniformConfig.high_load.mps_train,
                        port=run_comm.UniformConfig.port,
                        **system_config)
        run_comm.run(system, workload, server_model_config, 
                     "ablation-infer-idle-time-uniform", f"{max_infer_idle_ms}ms")
        

# MARK: cold cache cap
for cold_cache_min_cap, cold_cache_max_cap in cold_cache_cap_list:
    system_config = {
        'mode' : System.ServerMode.ColocateL1,
        'use_sta' : True, 
        'mps' : True, 
        'use_xsched' : True, 
        'has_warmup' : True,
        'ondemand_adjust' : True,
        'cuda_memory_pool_gb' : "13.5",
        'train_memory_over_predict_mb' : 1500,
        # 'infer_model_max_idle_ms' : 5000,
        'infer_model_max_idle_ms' : 500,
        'cold_cache_ratio': 0.5, 
        'cold_cache_min_capability_nbytes': int(cold_cache_min_cap * 1024 * 1024 * 1024),
        'cold_cache_max_capability_nbytes': int(cold_cache_max_cap * 1024 * 1024 * 1024),
    }
        
    with mps_thread_percent(run_comm.UniformConfig.high_load.mps_infer):
        client_model_list, server_model_config = InferModel.get_multi_model(
                run_comm.UniformConfig.model_list, run_comm.UniformConfig.num_model, 1)
        workload = run_comm.uniform(rps=run_comm.UniformConfig.high_load.rps, 
                                    client_model_list=client_model_list, infer_only=False,
                                    train_epoch=run_comm.get_train_epoch(run_comm.UniformConfig.train_epoch_time, run_comm.UniformConfig.duration))
        system = System(train_mps_thread_percent=run_comm.UniformConfig.high_load.mps_train,
                        port=run_comm.UniformConfig.port,
                        **system_config)
        run_comm.run(system, workload, server_model_config, 
                     "ablation-cold-cache-cap-uniform", f"{cold_cache_min_cap}GB-{cold_cache_max_cap}GB")


