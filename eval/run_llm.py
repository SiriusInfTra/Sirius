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
parser.add_argument('--infer-only', action='store_true')
parser.add_argument('--uniform-v2', action='store_true', help='Run uniform-v2')
parser.add_argument('--burstgpt', action='store_true', help='Run burstgpt')
parser.add_argument('--burstgpt-rps', type=int, help='BurstGPT RPS', required=False)
parser.add_argument('--train-mps-pct', type=int, help='Train MPS Pct', required=False)
parser.add_argument('--colsys-without-train', action='store_true', help='Run colsys without train') 

args = parser.parse_args()

run_colsys = False
run_um_mps = False
run_task_switch = False
run_static_partition = False
run_static_partition_I = False
run_static_partition_F = False
run_infer_only = False

colsys_without_train = False

enable_burstgpt = False
enable_uniform_v2 = False

uniform_v2_wkld_types = [
    # 'NormalB',
    # 'Normal_LogNormal_A'
    'Normal_LogNormal_LLM_D'
    # 'Normal_LogNormal_LLM_C'
    # 'Normal_Markov_LogNormal_LLM_AC'
]

run_comm.UniformConfig_v2.train_batch_size = 400
run_comm.UniformConfig_v2.interval_sec = 20

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
if args.infer_only:
    run_infer_only = True

if args.burstgpt:
    enable_burstgpt = True
if args.uniform_v2:
    enable_uniform_v2 = True

if args.colsys_without_train:
    colsys_without_train = True

if args.burstgpt_rps:
    run_comm.BurstGPTConfig.max_rps = args.burstgpt_rps

# LLM Workload
# llm_model = InferModel.Llama2_7B_HF
llm_model = InferModel.Llama2_13B_HF

# llm_max_model_len = 16000
llm_max_model_len = InferModel.get_llm_max_model_length(llm_model)

# Other config
run_comm.BurstGPTConfig.model_list = [llm_model]
port = run_comm.get_unique_port()

train_mps_pct = None
if args.train_mps_pct:
    train_mps_pct = args.train_mps_pct
    print('Train MPS Pct:', train_mps_pct)
print(run_colsys)


def get_cuda_memory_pool_gb():
    if llm_model == InferModel.Llama3_8B_Inst:
        return '58'
    elif llm_model == InferModel.Llama2_7B_HF:
        return '62'
    elif llm_model == InferModel.Llama2_13B_HF:
        return '51'
    else:
        raise ValueError(f"Unknown model: {llm_model}")
    
def get_cuda_memory_pool_gb_sp_f():
    if llm_model == InferModel.Llama3_8B_Inst:
        return '30'
    elif llm_model == InferModel.Llama2_7B_HF:
        return '32'
    elif llm_model == InferModel.Llama2_13B_HF:
        return '25'
    else:
        raise ValueError(f"Unknown model: {llm_model}")
    
def get_cuda_memory_pool_gb_sp_i():
    if llm_model == InferModel.Llama3_8B_Inst:
        return '44'
    elif llm_model == InferModel.Llama2_7B_HF:
        return '46'
    elif llm_model == InferModel.Llama2_13B_HF:
        return '37.5'
    else:
        raise ValueError(f"Unknown model: {llm_model}")



# MARK: colsys
if run_colsys:
    # Implement colsys logic
    system_config = {
        'mode': System.ServerMode.ColocateL1,
        'use_sta': True,
        'mps': True,
        'skip_set_mps_thread_percent': True,
        'use_xsched': True,
        'has_warmup': True,
        # 'cuda_memory_pool_gb': '58', # LLama3-8B
        # 'cuda_memory_pool_gb': '62', # LLama2-7B
        'cuda_memory_pool_gb': get_cuda_memory_pool_gb(),
        'train_memory_over_predict_mb': 3000,
        # 'infer_model_max_idle_ms': 5000,
        # 'cold_cache_ratio': 0.5,
        'dynamic_sm_partition': True,
        'serving_llm': True,
        'llm_model_name': llm_model,
        'llm_max_model_len': llm_max_model_len,
        'llm_show_gen_result': False,
    }
    if enable_burstgpt:
        with mps_thread_percent(None):
            workload = run_comm.burstgpt(infer_only=colsys_without_train)
            system = System(port=port, train_mps_thread_percent=train_mps_pct,
                            **system_config)
            run_comm.run(system, workload, None, 
                        f"burstgpt-{runner.get_num_gpu()}gpu", 
                        f"colsys")
    
    if enable_uniform_v2:
        for wkld_type in uniform_v2_wkld_types:
            client_model_list, _ = InferModel.get_multi_model([llm_model], 1, 1)
            workload = run_comm.uniform_v2(wkld_type, client_model_list, 
                                           infer_only=colsys_without_train)
            system = System(port=port, **system_config)
            run_comm.run(system, workload, None,
                         f'{wkld_type}-{runner.get_num_gpu()}gpu', 
                         f'colsys')


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
# Implement static-partition logic
for tag, item in {
    'I': ({
        'mode': System.ServerMode.Normal,
        'use_sta': False, # not using kv-cache pool
        'mps': True,
        'skip_set_mps_thread_percent': True,
        'use_xsched': False,
        'has_warmup': True,
        'dynamic_sm_partition': False,
        'cuda_memory_pool_gb': get_cuda_memory_pool_gb_sp_i(),
        'use_sta_train': False,
        'serving_llm': True,
        'llm_model_name': llm_model,
        'llm_max_model_len': llm_max_model_len,
        'llm_show_gen_result': False,
    }, {
        # 'train_batch_size': 100
        'train_batch_size': 8
    }),
    'F': ({
        'mode': System.ServerMode.Normal,
        'use_sta': False, # not using kv-cache pool
        'mps': True,
        'skip_set_mps_thread_percent': True,
        'use_xsched': False,
        'has_warmup': True,
        'dynamic_sm_partition': False,
        'cuda_memory_pool_gb': get_cuda_memory_pool_gb_sp_f(),
        'use_sta_train': False,
        'serving_llm': True,
        'llm_model_name': llm_model,
        'llm_max_model_len': llm_max_model_len,
        'llm_show_gen_result': False,
    }, {
        # 'train_batch_size': 180
        'train_batch_size': 32
    }),
}.items():
    if not run_static_partition:
        break
    if tag == 'F' and not run_static_partition_F:
        continue
    if tag == 'I' and not run_static_partition_I:
        continue

    system_config, workload_config = item
    if enable_burstgpt:
        with mps_thread_percent(None):
            workload = run_comm.burstgpt(
                infer_only=False,
                train_batch_size=workload_config['train_batch_size'])
            system = System(port=port, train_mps_thread_percent=train_mps_pct,
                            **system_config)
            run_comm.run(system, workload, workload_config,
                        f"burstgpt-{runner.get_num_gpu()}gpu", 
                        f"static-partition-{tag}")

    if enable_uniform_v2:
        for wkld_type in uniform_v2_wkld_types:
            with mps_thread_percent(None):
                client_model_list, _ = InferModel.get_multi_model([llm_model], 1, 1)
                workload = run_comm.uniform_v2(
                    wkld_type, client_model_list, infer_only=False,
                    train_batch_size=workload_config['train_batch_size']
                )
                system = System(port=port, train_mps_thread_percent=train_mps_pct,
                                **system_config)
                run_comm.run(system, workload, workload_config,
                            f'{wkld_type}-{runner.get_num_gpu()}gpu', 
                            f'static-partition-{tag}')


# MARK: Task Switch
if run_task_switch:
    pass


# MARK: UM MPS
if run_um_mps:
    # Implement UM+MPS logic
    system_config = {
        'mode': System.ServerMode.Normal,
        'use_sta': False,
        # 'cuda_memory_pool_gb': '58', # LLama3-8B
        # 'cuda_memory_pool_gb': '62', # LLama2-7B
        'cuda_memory_pool_gb': get_cuda_memory_pool_gb(),
        'mps': True,
        'skip_set_mps_thread_percent': False,
        'use_xsched': False,
        'has_warmup': True,
        'use_triton': True,
        'dynamic_sm_partition': False,
    }
    if enable_burstgpt:
        with um_mps(None):
            workload = run_comm.burstgpt(infer_only=False)
            system = System(port=port, 
                            train_mps_thread_percent=train_mps_pct,
                            **system_config)
            run_comm.run(system, workload, None,
                        f"burstgpt-{runner.get_num_gpu()}gpu", 
                        f"um-mps")


# MARK: infer only
if run_infer_only:
    system_config = {
        'mode': System.ServerMode.Normal,
        'use_sta': False,
        # 'cuda_memory_pool_gb': '58', # LLama3-8B
        # 'cuda_memory_pool_gb': '62', # LLama2-7B
        'cuda_memory_pool_gb': get_cuda_memory_pool_gb(),
        'mps': True,
        'skip_set_mps_thread_percent': True,
        'use_xsched': False,
        'has_warmup': True,
        'serving_llm': True,
        'llm_model_name': llm_model,
        'llm_max_model_len': llm_max_model_len,
        'llm_show_gen_result': False,
    }
    if enable_burstgpt:
        workload = run_comm.burstgpt(infer_only=True)
        system = System(port=port, **system_config)
        run_comm.run(system, workload, None,
                    f"burstgpt-{runner.get_num_gpu()}gpu", f"infer-only")

    if enable_uniform_v2:
        for wkld_type in uniform_v2_wkld_types:
            client_model_list, _ = InferModel.get_multi_model([llm_model], 1, 1)
            workload = run_comm.uniform_v2(wkld_type, client_model_list, infer_only=True)
            system = System(port=port, **system_config)
            run_comm.run(system, workload, None,
                         f'{wkld_type}-{runner.get_num_gpu()}gpu', f'infer-only')