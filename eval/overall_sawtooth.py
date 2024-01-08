from runner import *

set_global_seed(42)

use_time_stamp = True

run_colsys = True
run_um_mps = True
run_task_switch = True
run_infer_only = True

num_model = 36

def sawtooth(rps, infer_only=True):
    workload = HyperWorkload(concurrency=2048, duration=140, delay_before_infer=30,
                             warmup=5, delay_after_warmup=5, delay_before_profile=5)
    InferModel.reset_model_cnt()
    if not infer_only:
        workload.set_train_workload(train_workload=TrainWorkload('resnet', 15, 96))
    workload.set_infer_workloads(MicrobenchmarkInferWorkload(
        model_list=InferModel.get_model_list("resnet152", num_model),
        max_request_sec=rps, interval_sec=10, duration=65, 
        rps_fn=lambda i, rps : rps if i % 2 == 0 else 0,
    ))
    return workload

def run(system: System, workload: HyperWorkload, num_worker: int, tag: str):
    infer_model_config = System.InferModelConfig(f"resnet152[{num_model}]", "resnet152-b1", 1, num_worker)
    system.launch("overall-sawtooth", tag, time_stamp=use_time_stamp,
                  infer_model_config=infer_model_config)
    workload.launch_workload(system)
    system.stop()
    system.draw_memory_usage()
    system.draw_trace_cfg()
    time.sleep(5)


if run_colsys:
    # colsys heavy
    with mps_thread_percent(60):
        workload = sawtooth(rps=100, infer_only=False)
        system = System(mode=System.ServerMode.ColocateL1, use_sta=True, mps=True, use_xsched=True, 
                        cuda_memory_pool_gb="13.5", ondemand_adjust=True, train_memory_over_predict_mb=1500,
                        train_mps_thread_percent=40, infer_model_max_idle_ms=4000)
        run(system, workload, 0, "colsys-heavy")
    # # colsys light
    with mps_thread_percent(30):
        workload = sawtooth(rps=10, infer_only=False)
        system = System(mode=System.ServerMode.ColocateL1, use_sta=True, mps=True, use_xsched=True, 
                        cuda_memory_pool_gb="13.5", ondemand_adjust=True, train_memory_over_predict_mb=1500,
                        train_mps_thread_percent=70, infer_model_max_idle_ms=4000)
        run(system, workload, 0, "colsys-light")


if run_um_mps:
    # um+mps heavy
    with um_mps(60):
        workload = sawtooth(rps=100, infer_only=False)
        system = System(mode=System.ServerMode.Normal, use_sta=False, mps=True, use_xsched=False,
                        train_mps_thread_percent=40)
        run(system, workload, 1, "um-mps-heavy")
    # um+mps light
    with um_mps(30):
        workload = sawtooth(rps=10, infer_only=False)
        system = System(mode=System.ServerMode.Normal, use_sta=False, mps=True, use_xsched=False,
                        train_mps_thread_percent=70)
        run(system, workload, 1, "um-mps-light")


if run_task_switch:
    # task switch heavy
    workload = sawtooth(rps=100, infer_only=False)
    system = System(mode=System.ServerMode.TaskSwitchL1, use_sta=True, mps=False, use_xsched=False,
                    cuda_memory_pool_gb="13", train_memory_over_predict_mb=1500)
    run(system, workload, 0, "task-switch-heavy")
    # task switch light
    workload = sawtooth(rps=10, infer_only=False)
    system = System(mode=System.ServerMode.TaskSwitchL1, use_sta=True, mps=False, use_xsched=False,
                    cuda_memory_pool_gb="13", train_memory_over_predict_mb=1500)
    run(system, workload, 0, "task-switch-light")


if run_infer_only:
    # infer only heavy
    with mps_thread_percent(60):
        workload = sawtooth(rps=100, infer_only=True)
        system = System(mode=System.ServerMode.Normal, use_sta=False, mps=False, use_xsched=False)
        run(system, workload, 1, "infer-only-heavy")
    with mps_thread_percent(30):
        # infer only light
        workload = sawtooth(rps=10, infer_only=True)
        system = System(mode=System.ServerMode.Normal, use_sta=False, mps=False, use_xsched=False)
        run(system, workload, 1, "infer-only-light")




