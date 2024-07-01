from runner import *
set_global_seed(42)

import workload_collections as wkld_coll
import run_comm


run_comm.use_time_stamp = False
run_comm.retry_if_fail = True
run_comm.retry_limit = 1


intervals = [
    # 1, 5, 
    10, 20, 30
]

durations = [
    120, 180, 240, 300
]


for interval in intervals:
    for duration in durations:
        run_comm.UniformConfig_v2.interval_sec = interval
        run_comm.UniformConfig_v2.duration = duration

        for wkld_type in ["NormalA", "NormalB", "Normal_Markov_LogNormal_AC"]:

            # system_config = {
            #     'mode': System.ServerMode.ColocateL1,
            #     'use_sta': True,
            #     'mps': True,
            #     'use_xsched': True,
            #     'dynamic_sm_partition': True,
            #     'ondemand_adjust': True,
            #     'cuda_memory_pool_gb': "13",
            #     'train_memory_over_predict_mb': 1500,
            #     'infer_model_max_idle_ms': 5000,
            #     'cold_cache_ratio': 0.5, 
            #     'cold_cache_min_capability_nbytes': int(1.5 * 1024 * 1024 * 1024),
            #     'cold_cache_max_capability_nbytes': int(2 * 1024 * 1024 * 1024),
            # }
            # with mps_thread_percent(None):
            #     client_model_list, server_model_config = InferModel.get_multi_model(
            #         run_comm.UniformConfig_v2.model_list, run_comm.UniformConfig_v2.num_model, 1)
            #     workload = run_comm.uniform_v2(wkld_type, client_model_list, infer_only=False)
            #     system = System(port=run_comm.UniformConfig_v2.port,
            #                     **system_config)
            #     run_comm.run(system, workload, server_model_config,
            #                 "try-interval", f"colsys-{wkld_type}-I{interval}-D{duration}")
                

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
                with mps_thread_percent(None):
                    client_model_list, server_model_config = InferModel.get_multi_model(
                        run_comm.UniformConfig_v2.model_list, run_comm.UniformConfig_v2.num_model, 1)
                    workload = run_comm.uniform_v2(wkld_type, client_model_list, infer_only=False,
                                                train_batch_size=workload_config['train_batch_size'],
                                                train_epoch_time=workload_config['epoch_time'])
                    system = System(port=run_comm.UniformConfig_v2.port,
                                    **system_config)
                    run_comm.run(system, workload, server_model_config,
                                "try-interval", f"static-partition-{tag}-{wkld_type}-I{interval}-D{duration}")