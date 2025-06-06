import os
import argparse

import torch
from runner import *
set_global_seed(42)
from dataclasses import dataclass
import workload_collections as wkld_coll
import run_comm

# run_comm.UniformConfig_v2.train_model = 'swin_b_ddp'
run_comm.UniformConfig_v2.train_batch_size = 32
run_comm.UniformConfig_v2.train_global_batch_size = 512
run_comm.UniformConfig_v2.duration = 60

run_comm.UniformConfig_v2.real_data = True
run_comm.UniformConfig_v2.model_list = ['resnet152']
run_comm.UniformConfig_v2.num_model = 8
if torch.cuda.device_count() > 1:
    run_comm.UniformConfig_v2.train_model += "_ddp"
    run_comm.UniformConfig_v2.wait_train_setup_sec = 60
dynamic_sm_partition = False
skip_set_mps_pct = False
wkld_type = 'NormalA'

system_config = {
    'mode' : System.ServerMode.Normal,
    'use_sta' : True, 
    'mps' : True, 
    'skip_set_mps_thread_percent': skip_set_mps_pct,
    'use_xsched' : False,
    'has_warmup' : True,
    'ondemand_adjust' : True,
    'cuda_memory_pool_gb' : "8.0",
    'train_memory_over_predict_mb' : 1500,
    'infer_model_max_idle_ms' : 5000,
    'max_live_minute' : 10 + 7200 // 60,
    'cold_cache_ratio': 0, 
    'dynamic_sm_partition': dynamic_sm_partition,
}

with mps_thread_percent(None):
    client_model_list, server_model_config = InferModel.get_multi_model(
        run_comm.UniformConfig_v2.model_list, run_comm.UniformConfig_v2.num_model, 1)
    workload = run_comm.uniform_v2(wkld_type, client_model_list, infer_only=False)
    system = System(port=run_comm.UniformConfig_v2.port, use_triton=True, **system_config)
    run_comm.run(system, workload, server_model_config, 
                "overall-uniform-v2", f'sirius-{wkld_type}')