from runner import *

set_global_seed(42)

use_time_stamp = True

run_colsys = True
run_um_mps = True
run_task_switch = True

def smooth(rps, num_model, infer_only=True):
    workload = HyperWorkload(concurrency=2048, duration=140, delay_before_infer=30,
                            warmup=5, delay_after_warmup=5, delay_before_profile=5)
    InferModel.reset_model_cnt()
    if not infer_only:
        workload.set_train_workload(train_workload=TrainWorkload('resnet', 15, 96))
    workload.set_infer_workloads(MicrobenchmarkInferWorkload(
        model_list=InferModel.get_model_list("resnet152", num_model),
        max_request_sec=rps, interval_sec=1, duration=65
    ))
    return workload


def run(system: System, workload: HyperWorkload, num_worker: int, num_model: int, tag: str):
    infer_model_config = System.InferModelConfig(f"resnet152[{num_model}]", "resnet152-b1", 1, num_worker)
    system.launch("memory-pressure", tag, time_stamp=use_time_stamp,
                  infer_model_config=infer_model_config)
    workload.launch_workload(system)
    system.stop()
    time.sleep(5)
    system.draw_memory_usage()
    system.draw_trace_cfg()


# for num_model in [32, 36]:
for num_model in [8, 16, 24, 32, 36]:
    if run_colsys:
        with mps_thread_percent(60):
            hyper_workload = smooth(rps=100, num_model=num_model, infer_only=False)
            system = System(mode=System.ServerMode.ColocateL1, use_sta=True, mps=True, use_xsched=True, has_warmup=True,
                    cuda_memory_pool_gb="13.5", ondemand_adjust=True, train_memory_over_predict_mb=1500,
                    train_mps_thread_percent=40, infer_model_max_idle_ms=4000)
            run(system, hyper_workload, 0, num_model, f"colsys-{num_model}")

    if run_um_mps:
        with um_mps(60):
            hyper_workload = smooth(rps=100, num_model=num_model, infer_only=False)
            system = System(mode=System.ServerMode.Normal, use_sta=False, mps=True, use_xsched=False, has_warmup=True,
                            train_mps_thread_percent=40)
            run(system, hyper_workload, 1, num_model,f"um-mps-{num_model}")

    if run_task_switch:
        hyper_workload = smooth(rps=100, num_model=num_model, infer_only=False)
        system = System(mode=System.ServerMode.TaskSwitchL1, use_sta=True, mps=False, use_xsched=False, has_warmup=True,
                        cuda_memory_pool_gb="13", train_memory_over_predict_mb=1500)
        run(system, hyper_workload, 0, num_model, f"task-switch-{num_model}")
