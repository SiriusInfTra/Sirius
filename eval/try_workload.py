from runner import *
set_global_seed(42)

import workload_collections as wkld_coll
import run_comm


run_comm.use_time_stamp = True
run_comm.retry_if_fail = True
run_comm.retry_limit = 3
run_comm.fake_launch = False

run_comm.UniformConfig_v2.duration = 300
run_comm.UniformConfig_v2.interval_sec = 20
run_comm.SkewedConfig_v2.duration = 300
run_comm.SkewedConfig_v2.interval_sec = 20


run_colsys = True
run_static_partition = True

normal_wklds = wkld_coll.get_all_normal_workload(name_filter=r'LogNormal|Weibull')
# wklds = normal_wklds
skew_wklds = wkld_coll.get_all_skew_workload(name_filter=r'LogNormal|Weibull')
wklds = {**normal_wklds, **skew_wklds}

# wklds = skew_wklds
# wklds = wkld_coll.get_all_normal_workload(name_filter=r'Normal_Markov_Weibull')

# wklds = wkld_coll.get_all_skew_workload(name_filter=r'Markov_Weibull_CD')

print('try workload: ', wklds.keys())

# for wkld_name, wkld in wklds.items():
if run_colsys:
    system_config = {
        'mode': System.ServerMode.ColocateL1,
        'use_sta': True,
        'mps': True,
        'skip_set_mps_thread_percent': True,
        'use_xsched': True,
        'dynamic_sm_partition': True,
        'ondemand_adjust': True,
        'cuda_memory_pool_gb': "13",
        'train_memory_over_predict_mb': 1500,
        'infer_model_max_idle_ms': 5000,
        'cold_cache_ratio': 0.5, 
        'cold_cache_min_capability_nbytes': int(1.5 * 1024 * 1024 * 1024),
        'cold_cache_max_capability_nbytes': int(2 * 1024 * 1024 * 1024),
    }

    for wkld_name, wkld in normal_wklds.items():
        # with mps_thread_percent(50):
        with mps_thread_percent(None):
            client_model_list, server_model_config = InferModel.get_multi_model(
                run_comm.UniformConfig_v2.model_list, run_comm.UniformConfig_v2.num_model, 1)
            workload = run_comm.uniform_v2(wkld, client_model_list, infer_only=False)
            system = System(port=run_comm.UniformConfig_v2.port,
                            **system_config)
            run_comm.run(system, workload, server_model_config,
                        "try-workload", f"colsys-{wkld_name}")
            
    for wkld_name, wkld in skew_wklds.items():
        # with mps_thread_percent(50):
        with mps_thread_percent(None):
            client_model_list, server_model_config = InferModel.get_multi_model(
                run_comm.SkewedConfig_v2.model_list, run_comm.SkewedConfig_v2.num_model, 1)
            workload = run_comm.skewed_v2(wkld, client_model_list, infer_only=False)
            system = System(port=run_comm.SkewedConfig_v2.port,
                            **system_config)
            run_comm.run(system, workload, server_model_config,
                        "try-workload", f"colsys-{wkld_name}")


if run_static_partition:
    for tag, item in {
        'F': ({
            'mode' : System.ServerMode.Normal,
            'use_sta': True,
            'mps': True,
            'skip_set_mps_thread_percent': True,
            'use_xsched': True,
            'dynamic_sm_partition': True,
            'has_warmup': True,
            'max_warm_cache_nbytes': int(5.5 * 1024 ** 3),
            'cuda_memory_pool_gb': '7',
            'use_sta_train': False
        }, {'train_batch_size': 32, 'epoch_time': 5.5}),
        'I': ({
            'mode' : System.ServerMode.Normal,
            'use_sta': True,
            'mps': True,
            'skip_set_mps_thread_percent': True,
            'use_xsched': True,
            'dynamic_sm_partition': True,
            'has_warmup': True,
            'max_warm_cache_nbytes': int(9 * 1024 ** 3),
            'cuda_memory_pool_gb': '10.5',
            'use_sta_train': False
        }, {'train_batch_size': 8, 'epoch_time': 14.5}), 
    }.items():
        system_config, workload_config = item

        for wkld_name, wkld in normal_wklds.items():
            # with mps_thread_percent(50):
            with mps_thread_percent(None):
                client_model_list, server_model_config = InferModel.get_multi_model(
                    run_comm.UniformConfig_v2.model_list, run_comm.UniformConfig_v2.num_model, 1)
                workload = run_comm.uniform_v2(wkld, client_model_list, infer_only=False,
                                            train_batch_size=workload_config['train_batch_size'],
                                            train_epoch_time=workload_config['epoch_time'])
                system = System(port=run_comm.UniformConfig_v2.port,
                                **system_config)
                run_comm.run(system, workload, server_model_config,
                            "try-workload", f"static-partition-{tag}-{wkld_name}")
        
        for wkld_name, wkld in skew_wklds.items():
            # with mps_thread_percent(50):
            with mps_thread_percent(None):
                client_model_list, server_model_config = InferModel.get_multi_model(
                    run_comm.SkewedConfig_v2.model_list, run_comm.SkewedConfig_v2.num_model, 1)
                workload = run_comm.skewed_v2(wkld, client_model_list, infer_only=False,
                                              train_batch_size=workload_config['train_batch_size'],
                                              train_epoch_time=workload_config['epoch_time'])
                system = System(port=run_comm.SkewedConfig_v2.port,
                                **system_config)
                run_comm.run(system, workload, server_model_config,
                             "try-workload", f"static-partition-{tag}-{wkld_name}")



# ==========================================================================
# ==========================================================================

# import os, sys, re
# import pathlib
# import pandas as pd


# def parse_test_unit_name(name:str):
#     m = re.match(r'(.*)-((Normal|Skew).*)', name)
#     if m:
#         return m.group(1), m.group(2)
#     else:
#         raise ValueError(f'Invalid test unit name: {name}')


# def get_perf(log_dir:pathlib.Path):
#     try:
#         workload_log = log_dir / 'workload-log'
#         train_thpt_log = log_dir / 'train_thpt'
#         if not workload_log.exists() or not train_thpt_log.exists():
#             log_dir = pathlib.Path(str(log_dir) + '-retry-0')
#             workload_log = log_dir / 'workload-log'
#             train_thpt_log = log_dir / 'train_thpt'
#         with open(workload_log, 'r') as f:
#             p99 = re.search(r'p99 ([^ ]+) ', f.readlines()[2])
#         with open(train_thpt_log, 'r') as f:
#             thpt = re.search(r'.*epoch e2e thpt: ([^ ]+) .*', f.read())
#         return float(p99.group(1)), float(thpt.group(1))
#     except Exception:
#         return None, None


# # log_root = 'log/try-workload-20240630-2330'
# # log_root = 'log/try-workload-20240703-0901'
# log_root = 'log/try-workload-20240704-1001'

# df = pd.DataFrame({
#     'Unit': [],
#     'colsys-P99': [],
#     'SP-F-P99': [],
#     'SP-I-P99': [],

#     'colsys-Thpt': [],
#     'SP-F-Thpt': [],
#     'SP-I-Thpt': [],
# })

# test_units = [unit.name for unit in pathlib.Path(log_root).glob('*') 
#               if 'retry' not in unit.name]
# test_units = [parse_test_unit_name(unit)[1] for unit in test_units]
# test_units = list(set(test_units))

# for unit in test_units:
#     colsys_p99, colsys_thpt = get_perf(pathlib.Path(log_root) / f'colsys-{unit}')
#     sp_f_p99, sp_f_thpt = get_perf(pathlib.Path(log_root) / f'static-partition-F-{unit}')
#     sp_i_p99, sp_i_thpt = get_perf(pathlib.Path(log_root) / f'static-partition-I-{unit}')
#     df = pd.concat([df, pd.DataFrame(
#         [[unit, colsys_p99, sp_f_p99, sp_i_p99, colsys_thpt, sp_f_thpt, sp_i_thpt]], 
#         columns=df.columns
#     )], ignore_index=True)

# df = df.sort_values(by=['Unit'])

# print(df.to_string())
    