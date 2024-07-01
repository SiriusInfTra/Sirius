from runner import *
set_global_seed(42)

import workload_collections as wkld_coll
import run_comm


run_comm.use_time_stamp = False
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
        with mps_thread_percent(None):
            client_model_list, server_model_config = InferModel.get_multi_model(
                run_comm.UniformConfig_v2.model_list, run_comm.UniformConfig_v2.num_model, 1)
            workload = run_comm.uniform_v2(wkld, client_model_list, infer_only=False)
            system = System(port=run_comm.UniformConfig_v2.port,
                            **system_config)
            run_comm.run(system, workload, server_model_config,
                        "try-workload", f"colsys-{wkld_name}")
            
    for wkld_name, wkld in skew_wklds.items():
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
