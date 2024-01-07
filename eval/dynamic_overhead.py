from runner import *

set_global_seed(42)

use_time_stamp = True

def azure(rps, infer_only=False):
    workload = HyperWorkload(concurrency=2048, duration=140, delay_before_infer=30,
                            warmup=5, delay_after_warmup=5)
    InferModel.reset_model_cnt()
    if not infer_only:
        workload.set_train_workload(train_workload=TrainWorkload('resnet', 15, 96))
    workload.set_infer_workloads(AzureInferWorkload(
        AzureInferWorkload.TRACE_D01,
        model_list=InferModel.get_model_list("resnet152", 36),
        max_request_sec=rps, interval_sec=1, period_num=60, func_num=16 * 1000, 
    ))
    
    return workload


def run(system: System, workload: HyperWorkload, num_worker: int, tag: str):
    infer_model_config = System.InferModelConfig("resnet152[36]", "resnet152-b1", 1, num_worker)
    system.launch("dynamic-overhead", tag, time_stamp=use_time_stamp,
                  infer_model_config=infer_model_config)
    workload.launch_workload(system)
    system.stop()
    time.sleep(5)
    system.draw_memory_usage()
    system.draw_trace_cfg()


for max_idle_ms in [500, 1000, 2000, 4000, 5000, 8000, 10000, 1000*1000]:
    with mps_thread_percent(60):
        workload = azure(rps=100, infer_only=False)
        system = System(mode=System.ServerMode.ColocateL1, use_sta=True, mps=True, use_xsched=True, 
                cuda_memory_pool_gb="13.5", ondemand_adjust=True, train_memory_over_predict_mb=1500,
                train_mps_thread_percent=40, memory_pool_policy=System.MemoryPoolPolicy.NextFit,
                infer_model_max_idle_ms=max_idle_ms, has_warmup=True)
        tag = f"{max_idle_ms}" if max_idle_ms < 1000*1000 else "inf"
        run(system, workload, 0, tag)