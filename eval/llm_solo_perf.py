import os
import argparse
from runner import *
set_global_seed(42)

from dataclasses import dataclass
import workload_collections as wkld_coll
import run_comm

parser = argparse.ArgumentParser(description='Run LLM')

# llm_model = InferModel.Llama3_8B_Inst
# llm_model = InferModel.Llama2_7B_HF
llm_model = InferModel.Llama2_13B_HF
llm_max_model_len = InferModel.get_llm_max_model_length(llm_model)
port = run_comm.get_unique_port()

system_config = {
    'mode': System.ServerMode.ColocateL1,
    'use_sta': False,
    # 'cuda_memory_pool_gb': '58',
    # 'cuda_memory_pool_gb': '62',
    'cuda_memory_pool_gb': '50',
    'mps': True,
    'skip_set_mps_thread_percent': True,
    'use_xsched': False,
    'has_warmup': True,
    'serving_llm': True,
    'llm_model_name': llm_model,
    'llm_max_model_len': llm_max_model_len,
    'llm_show_gen_result': False,
    'max_live_minute': 60,
}

system = System(port=port, **system_config)
workload = HyperWorkload(concurrency=1, 
                         duration=1000, 
                         warmup=5,
                         wait_warmup_done_sec=5,
                         wait_stable_before_start_profiling_sec=0)

InferModel.reset_model_cnt()
client_model_list, _ = InferModel.get_multi_model([llm_model], 1, 1)

system.launch("llm-solo-perf", f'{llm_model}', time_stamp=True,
              infer_model_config=None)
workload.launch_busy_loop(system, client_model_list)
system.stop()
