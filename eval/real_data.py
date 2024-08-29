import os
import argparse

import torch
from runner import *
set_global_seed(42)

from dataclasses import dataclass
import workload_collections as wkld_coll
import run_comm

# run_comm.UniformConfig_v2.train_model = 'swin_b_ddp'
run_comm.UniformConfig_v2.train_batch_size = 48
run_comm.UniformConfig_v2.train_global_batch_size = 1024
run_comm.UniformConfig_v2.duration = 7200
run_comm.UniformConfig_v2.real_data = True
if torch.cuda.device_count() > 1:
    run_comm.UniformConfig_v2.train_model += "_ddp"
    run_comm.UniformConfig_v2.wait_train_setup_sec = 60
dynamic_sm_partition = True
skip_set_mps_pct = False
wkld_type = 'NormalA'

system_config = {
    'mode' : System.ServerMode.ColocateL1,
    'use_sta' : True, 
    'mps' : True, 
    'skip_set_mps_thread_percent': skip_set_mps_pct,
    'use_xsched' : True,
    'has_warmup' : True,
    'ondemand_adjust' : True,
    'cuda_memory_pool_gb' : "12.5",
    'train_memory_over_predict_mb' : 1500,
    'infer_model_max_idle_ms' : 5000,
    'max_live_minute' : 10 + 7200 // 60,
    'cold_cache_ratio': 0.5, 
    'cold_cache_min_capability_nbytes': int(1.5 * 1024 * 1024 * 1024),
    'cold_cache_max_capability_nbytes': int(2 * 1024 * 1024 * 1024),
    'dynamic_sm_partition': dynamic_sm_partition,
}

with mps_thread_percent(None):
    client_model_list, server_model_config = InferModel.get_multi_model(
        run_comm.UniformConfig_v2.model_list, run_comm.UniformConfig_v2.num_model, 1)
    workload = run_comm.uniform_v2(wkld_type, client_model_list, infer_only=False)
    system = System(port=run_comm.UniformConfig_v2.port, **system_config)
    run_comm.run(system, workload, server_model_config, 
                "overall-uniform-v2", f'colsys-{wkld_type}')