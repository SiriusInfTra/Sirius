import os
import argparse
from runner import *
from dataclasses import dataclass
import run_comm

import argparse

run_comm.use_time_stamp = True
run_comm.retry_if_fail = False
run_comm.skip_fail = False

parser = argparse.ArgumentParser()
parser.add_argument('--retry-limit', type=int, default=0)
args = parser.parse_args()

if args.retry_limit > 0:
    run_comm.retry_if_fail = True
    run_comm.retry_limit = args.retry_limit
    run_comm.skip_fail = True


# run_comm.fake_launch = True

set_global_seed(42)

# MARK: Config
class UniformConfig:
    train_model = 'swin_b'
    train_batch_size = 72
    train_global_batch_size = 500 # not used, hard code for global batch size and dataset size
    train_dataset_size = 1000 
    train_epoch_time = 5.5 # used for predict number epoch

    model_list = [InferModel.ResNet152] 
    num_model = 96
    interval_sec = 60
    duration = None
    port = str(run_comm.get_unique_port())
    enable = True

    low_load = run_comm.LowLoad(enable=False)
    high_load = run_comm.HighLoad(enable=True)
    hybrid_load = run_comm.HybridLoad(enable=False)

delta = 4

num_model_to_request = [4, 4]
for i in range(23):
    num_model_to_request.append(num_model_to_request[-1] + delta)
# for i in range(23):
#     num_model_to_request.append(num_model_to_request[-1] - delta)
# num_model_to_request.append(delta)

print(num_model_to_request, len(num_model_to_request))
UniformConfig.duration = (len(num_model_to_request)-1) * UniformConfig.interval_sec

assert UniformConfig.num_model == max(num_model_to_request), \
    f"num_model {UniformConfig.num_model} != max(num_model_to_request) {max(num_model_to_request)}"


# MARK: Workload
## =========================================================== ##

def uniform(rps, client_model_list, infer_only=True, rps_fn=None, num_model_request_fn=None,
            train_model:str = UniformConfig.train_model, 
            train_epoch:int = int(UniformConfig.duration / UniformConfig.train_epoch_time + 5), 
            train_batch_size:int = UniformConfig.train_batch_size):
    workload = HyperWorkload(concurrency=2048,
                             warmup=5,
                             wait_warmup_done_sec=5,
                             wait_train_setup_sec=40 ,
                             wait_stable_before_start_profiling_sec=UniformConfig.interval_sec)
    InferModel.reset_model_cnt()
    if not infer_only:
        workload.set_train_workload(
            train_workload=TrainWorkload(train_model, train_epoch, train_batch_size))
    workload.set_infer_workloads(MicrobenchmarkInferWorkload_v1(
        model_list=client_model_list,
        interval_sec=UniformConfig.interval_sec, fix_request_sec=rps,
        rps_fn=rps_fn, num_request_model_fn=num_model_request_fn,
        equal_partition_rps=True,
        sequential_choose_model=True,
        duration=UniformConfig.duration + workload.infer_extra_infer_sec,
        verbose=True
    ))
    return workload


# MARK: COLSYS
system_config = {
    'mode' : System.ServerMode.ColocateL1,
    'use_sta' : True, 
    'mps' : True, 
    'skip_set_mps_thread_percent': True,
    'use_xsched' : True, 
    'has_warmup' : True,
    'ondemand_adjust' : True,
    'cuda_memory_pool_gb' : "13.0",
    'train_memory_over_predict_mb' : 1500,
    'infer_model_max_idle_ms' : 5000,
    'cold_cache_ratio': 0.5, 
    'cold_cache_min_capability_nbytes': int(1.5 * 1024 * 1024 * 1024),
    'cold_cache_max_capability_nbytes': int(2 * 1024 * 1024 * 1024),
    'dynamic_sm_partition': True,
}

with mps_thread_percent(UniformConfig.high_load.mps_infer):
    def num_model_request_fn(i, m):
        global num_model_to_request
        return num_model_to_request[i]
    

    client_model_list, server_model_config = InferModel.get_multi_model(
        UniformConfig.model_list, UniformConfig.num_model, 1)
    workload = uniform(rps=UniformConfig.high_load.rps, 
                       client_model_list=client_model_list, infer_only=False,
                       num_model_request_fn=num_model_request_fn)
    # workload = run_comm.uniform(rps=UniformConfig.high_load.rps, 
    #                    client_model_list=client_model_list, infer_only=False)
    system = System(train_mps_thread_percent=UniformConfig.high_load.mps_train,
                    port=UniformConfig.port,
                    max_live_minute=int(UniformConfig.duration/60 + 10),
                    **system_config)
    run_comm.run(system, workload, server_model_config, 
                 "memory-pressure", f"{UniformConfig.num_model}")