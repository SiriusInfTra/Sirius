from runner import *

set_global_seed(42)

def get_workload(rps, infer_only=False):
    workload = HyperWorkload(concurrency=2048, duration=140, 
                             warmup=5, wait_warmup_done_sec=5,
                             wait_train_setup_sec=0)
    InferModel.reset_model_cnt()
    workload.set_infer_workloads(MicrobenchmarkInferWorkload_v1(
        model_list=InferModel.get_model_list("resnet152", 32),
        max_request_sec=rps, interval_sec=5, duration=60, 
    ))
    return workload

def azure(rps, infer_only=True):
    workload = HyperWorkload(concurrency=2048, duration=140, 
                             warmup=5, wait_warmup_done_sec=5,
                             wait_train_setup_sec=30,)
    InferModel.reset_model_cnt()
    if not infer_only:
        workload.set_train_workload(train_workload=TrainWorkload('resnet', 15, 96))
    workload.set_infer_workloads(AzureInferWorkload(
        AzureInferWorkload.TRACE_D01,
        model_list=InferModel.get_model_list("resnet152", 32),
        max_request_sec=rps, interval_sec=1, period_num=60, func_num=16 * 1000, 
    ))
    
    return workload

def smooth(rps, infer_only=True):
    workload = HyperWorkload(concurrency=2048, duration=140,
                             warmup=5, wait_warmup_done_sec=5,
                             wait_train_setup_sec=30,)
    InferModel.reset_model_cnt()
    if not infer_only:
        workload.set_train_workload(train_workload=TrainWorkload('resnet', 15, 96))
    workload.set_infer_workloads(MicrobenchmarkInferWorkload_v1(
        model_list=InferModel.get_model_list("resnet152", 32),
        max_request_sec=rps, interval_sec=1, duration=60
    ))
    return workload

def run(system: System, workload: Workload, num_worker: int, tag: str):
    infer_model_config = System.InferModelConfig("resnet152[32]", "resnet152-b1", 1, num_worker)
    system.launch("try-infer-mps", tag, time_stamp=False,
                  infer_model_config=infer_model_config)
    workload.launch_workload(system)
    system.stop()
    time.sleep(1)


# infer only heavy
hyper_workload = azure(rps=100, infer_only=True)
system = System(mode=System.ServerMode.Normal, use_sta=False, mps=False, use_xsched=False)
run(system, hyper_workload, 1, "heavy-100")
# infer only light
hyper_workload = azure(rps=10, infer_only=True)
system = System(mode=System.ServerMode.Normal, use_sta=False, mps=False, use_xsched=False)
run(system, hyper_workload, 1, "light-100")

for mps_pct in [50, 60, 70, 80, 90]:
    with mps_thread_percent(mps_pct):
        hyper_workload = azure(rps=100, infer_only=True)
        system = System(mode=System.ServerMode.Normal, use_sta=False, mps=True, use_xsched=False)
        run(system, hyper_workload, 1, f"heavy-{mps_pct}")

for mps_pct in [10, 20, 30, 40, 50]:
    with mps_thread_percent(mps_pct):
        hyper_workload = azure(rps=10, infer_only=True)
        system = System(mode=System.ServerMode.Normal, use_sta=False, mps=True, use_xsched=False)
        run(system, hyper_workload, 1, f"light-{mps_pct}")




