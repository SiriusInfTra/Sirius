import os
import argparse
from runner import *
set_global_seed(42)

from dataclasses import dataclass
import workload_collections as wkld_coll
import run_comm

parser = argparse.ArgumentParser(description='Run LLM')
parser.add_argument('--colsys', action='store_true', help='Run colsys')
parser.add_argument('--um-mps', action='store_true', help='Run UM+MPS')
parser.add_argument('--task-switch', action='store_true', help='Run task switch')
parser.add_argument('--static-partition', action='store_true', help='Run static partition')
parser.add_argument('--static-partition-i', action='store_true', help='Run static partition I')
parser.add_argument('--static-partition-f', action='store_true', help='Run static partition F')

args = parser.parse_args()

run_colsys = False
run_um_mps = False
run_task_switch = False
run_static_partition = False
run_static_partition_I = False
run_static_partition_F = False

if args.colsys:
    run_colsys = True
if args.um_mps:
    run_um_mps = True
if args.task_switch:
    run_task_switch = True
if args.static_partition:
    run_static_partition = True
    run_static_partition_I = True
    run_static_partition_F = True
if args.static_partition_i:
    run_static_partition = True
    run_static_partition_I = True
if args.static_partition_f:
    run_static_partition = True
    run_static_partition_F = True

# LLM Workload
llm_model = "meta-llama/Llama-3.1-8B"
llm_max_seq_len = 512
llm_max_batch_size = 8

# Other config
port = run_comm.get_unique_port()


print(run_colsys)

# MARK: colsys
if run_colsys:
    # Implement colsys logic
    system_config = {
        'mode': System.ServerMode.ColocateL1,
        'use_sta': True,
        'mps': True,
        'skip_set_mps_thread_percent': False,
        'use_xsched': True,
        'has_warmup': True,
        'cuda_memory_pool_gb': '70',
        'train_memory_over_predict_mb': 1000,
        # 'infer_model_max_idle_ms': 5000,
        # 'cold_cache_ratio': 0.5,
        'dynamic_sm_partition': True,
        'serving_llm': True,
        'llm_model_name': llm_model,
        'llm_max_seq_len': llm_max_seq_len,
        'llm_max_batch_size': llm_max_batch_size,
    }
    workload = run_comm.llm(infer_only=True)
    system = System(**system_config)
    run_comm.run(system, workload, None, 
                 f"llm-{runner.get_num_gpu()}gpu", f"colsys")
    # ...additional colsys logic...


# MARK: task switch
if run_task_switch:
    # Implement task-switch logic
    system_config = {
        'mode': System.ServerMode.TaskSwitchL1,
        'use_sta': True,
        'mps': False,
        'use_xsched': True,
        'has_warmup': True,
        'cuda_memory_pool_gb': '13',
        'train_memory_over_predict_mb': 1500,
    }
    system = System(**system_config)
    # ...additional task-switch logic...


# MARK: static partition
if run_static_partition:
    # Implement static-partition logic
    system_config = {
        'mode': System.ServerMode.Normal,
        'use_sta': True,
        'mps': True,
        'skip_set_mps_thread_percent': False,
        'use_xsched': False,
        'has_warmup': True,
        'max_warm_cache_nbytes': int(9 * 1024 ** 3),
        'cuda_memory_pool_gb': '10.5',
        'use_sta_train': False
    }
    system = System(**system_config)
    # ...additional static-partition logic...


# MARK: UM MPS
if run_um_mps:
    # Implement UM+MPS logic
    system_config = {
        'mode': System.ServerMode.Normal,
        'use_sta': False,
        'mps': True,
        'skip_set_mps_thread_percent': False,
        'use_xsched': False,
        'has_warmup': True,
        'use_triton': True,
        'dynamic_sm_partition': False,
    }
    with mps_thread_percent(None):
        system = System(port=port, **system_config)


    # ...additional UM+MPS logic...

# ...existing code...
