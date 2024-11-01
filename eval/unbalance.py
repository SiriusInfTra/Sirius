import os
import argparse
import shutil
from runner import *
RunnerConfig._multi_gpu_scale_up_workload = False

set_global_seed(42)

from dataclasses import dataclass
import workload_collections as wkld_coll
import run_comm
import tempfile


def update_sever_config(server_model_config: str, num_workloads: int) -> str:
    for k in range(len(server_model_config)):
        def update(match):
            # 提取数字并转换为整数
            num = int(match.group(1))
            # 返回替换后的字符串
            return f'[{num * num_workloads}]'
        
        # 使用正则表达式查找并替换
        server_model_config[k] = re.sub(r'\[(\d+)\]', update, server_model_config[k])
    return server_model_config
    

def merge_workload(workload_list: list[HyperWorkload], 
                   trace_cfg_dir: str) -> HyperWorkload:
    model_def_text_list: list[list[str]] = []
    model_rep_text_list: list[list[str]]  = []
    for i, workload in enumerate(workload_list):
        trace_cfg = pathlib.Path(trace_cfg_dir) / f"trace-{i}.cfg"
        InferTraceDumper(workload.infer_workloads, trace_cfg).dump()
        with open(trace_cfg, 'r') as f:
            contents = f.readlines()
            model_def = contents.index('# model_id,model_name\n')
            model_rep = contents.index('# start_point,model_id\n')
            model_def_text = contents[model_def + 1:model_rep]
            model_rep_text = contents[model_rep + 1:]
            model_def_text_list.append(model_def_text)
            model_rep_text_list.append(model_rep_text)
            print(f'append {len(model_rep_text)}')
    for model_def_text in model_def_text_list[1:]:
        assert model_def_text == model_def_text_list[0]
    trace_cfg = pathlib.Path(trace_cfg_dir) / f"trace-merged.cfg"
    num_workloads = len(model_def_text_list)
    num_models = len(model_def_text_list[0])
    with open(trace_cfg, 'w') as f:
        f.write('# model_id,model_name\n')
        model_def_text_flat_list = []
        for j, model_def_text in enumerate(model_def_text_list):
            for line in model_def_text:
                model_id, model_name = line.strip().split(',')
                new_model_id = str(int(model_id) * num_workloads + j)
                if '-' in model_name:
                    model_basename, model_dup_id = model_name.split('-')
                else:
                    model_basename, model_dup_id = model_name, 0
                new_model_dup_id = str(int(model_dup_id) * num_workloads + j)
                if int(new_model_dup_id) > 0:
                    new_model_name = f"{model_basename}-{new_model_dup_id}"
                else:
                    new_model_name = model_basename
                model_def_text_flat_list.append(f"{new_model_id},{new_model_name}\n")
        model_def_text_flat_list.sort(key=lambda s: int(str(s).split(',')[0]))
        f.writelines(model_def_text_flat_list)
        f.write('# start_point,model_id\n')
        model_rep_text_flat_list = []
        for j, model_rep_text in enumerate(model_rep_text_list):
            print(f'write: {len(model_rep_text)}')
            for line in model_rep_text:
                start_point, model_id = line.strip().split(',')
                new_model_id = int(model_id) * num_workloads + j
                # print(num_models, j, model_id, '->', new_model_id)
                # print(model_id, '->', new_model_id)
                model_rep_text_flat_list.append(f"{start_point},{new_model_id}\n")
        model_rep_text_flat_list.sort(key=lambda s: float(str(s).split(',')[0]))
        f.writelines(model_rep_text_flat_list)
    workload_list[0].manual_trace_cfg = str(trace_cfg)
    return workload_list[0]

run_comm.UniformConfig_v2.train_model += "_ddp"
run_comm.SkewedConfig_v2.train_model += "_ddp"
run_comm.UniformConfig_v2.train_batch_size = 72 if not runner.is_four_gpu() else 66
run_comm.retry_limit = True
run_comm.retry_if_fail = 3

for train_adjust_balance in [
    True, 
    False
]:
    system_config = {
        'mode' : System.ServerMode.ColocateL1,
        'use_sta' : True, 
        'mps' : True, 
        'skip_set_mps_thread_percent': False,
        'use_xsched' : True,
        'has_warmup' : True,
        'ondemand_adjust' : True,
        'cuda_memory_pool_gb' : "13" if not runner.is_four_gpu() else "12.5",
        'train_memory_over_predict_mb' : 1500,
        'infer_model_max_idle_ms' : 5000,
        'cold_cache_ratio': 0.5, 
        'train_adjust_balance': train_adjust_balance,
        # 'cold_cache_min_capability_nbytes': int(0.5 * 1024 * 1024 * 1024),
        # 'cold_cache_max_capability_nbytes': int(1 * 1024 * 1024 * 1024),
        'cold_cache_min_capability_nbytes': int(1.5 * 1024 * 1024 * 1024),
        'cold_cache_max_capability_nbytes': int(2 * 1024  * 1024 * 1024),
        'dynamic_sm_partition': True,
        'train_adjust_batch_size_limit': 1,
    }
    client_model_list, server_model_config = InferModel.get_multi_model(
        run_comm.UniformConfig_v2.model_list, 40, 1)
    workload_list = [
        run_comm.uniform_v2(
            wkld_type, 
            client_model_list, 
            infer_only=False
        ) for wkld_type in ['NormalA', 'NormalB']
    ]

    system = System(port=run_comm.UniformConfig_v2.port, 
                    dump_adjust_info=False,
                    **system_config)
    tmpdir = 'unbal_tmp'
    if not os.path.exists(tmpdir):
        os.mkdir(tmpdir)
    print('Using temporary directory', tmpdir)
    workload = merge_workload(workload_list, tmpdir)
    server_model_config = update_sever_config(server_model_config, len(workload_list))

    run_comm.run(system, workload, server_model_config,
                    f"overall-uniform-v2-{runner.get_num_gpu()}gpu", 
                    'colsys-' + ('balance' if train_adjust_balance else 'imbalance'))