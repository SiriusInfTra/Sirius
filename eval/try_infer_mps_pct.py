from runner import *

set_global_seed(42)

def get_workload(rps, infer_only=False):
    workload = HyperWorkload(concurrency=2048, duration=140, delay_before_infer=0,
                            warmup=5, delay_after_warmup=5)
    InferModel.reset_model_cnt()
    workload.set_infer_workloads(MicrobenchmarkInferWorkload(
        model_list=InferModel.get_model_list("resnet152", 32),
        max_request_sec=rps, interval_sec=5, duration=60, 
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
workload = get_workload(rps=100, infer_only=True)
system = System(mode=System.ServerMode.Normal, use_sta=False, mps=False, use_xsched=False)
run(system, workload, 1, "heavy-100")
# infer only light
workload = get_workload(rps=10, infer_only=True)
system = System(mode=System.ServerMode.Normal, use_sta=False, mps=False, use_xsched=False)
run(system, workload, 1, "light-100")

for mps_pct in [50, 60, 70, 80, 90]:
    with mps_thread_percent(mps_pct):
        workload = get_workload(rps=100, infer_only=True)
        system = System(mode=System.ServerMode.Normal, use_sta=False, mps=True, use_xsched=False)
        run(system, workload, 1, f"heavy-{mps_pct}")

for mps_pct in [10, 20, 30, 40, 50]:
    with mps_thread_percent(mps_pct):
        workload = get_workload(rps=10, infer_only=True)
        system = System(mode=System.ServerMode.Normal, use_sta=False, mps=True, use_xsched=False)
        run(system, workload, 1, f"light-{mps_pct}")




