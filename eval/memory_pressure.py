from runner import *
from dataclasses import dataclass

set_global_seed(42)

use_time_stamp = True

run_colsys = False
run_um_mps = True
run_task_switch = True

# run_colsys = False
# run_um_mps = False
# run_task_switch = False

@dataclass
class SmoothConfig:
    model_list = [InferModel.ResNet152]
    heavy_rps = 100
    heavy_num_model = 60
    heavy_mps_infer = 60
    heavy_mps_train = 40
    heavy_try_mps_pct_list = [0]
    # heavy_try_mps_pct_list = [50, 60, 70, 80]

    light_rps = 10
    light_num_model = 32
    light_mps_infer = 30
    light_mps_train = 70
    light_try_mps_pct_list = [0]
    # light_try_mps_pct_list = [0, 20, 30, 40]

    num_model_list = [8, 16, 24, 32, 36]
    server_port = "18580"
    increase_time = 120
    
eval_config = SmoothConfig()

def smooth(rps, client_model_list, infer_only=True):
    workload = HyperWorkload(concurrency=2048, duration=eval_config.increase_time + 140, delay_before_infer=30,
                            warmup=5, delay_after_warmup=5, delay_before_profile=5)
    InferModel.reset_model_cnt()
    if not infer_only:
        workload.set_train_workload(train_workload=TrainWorkload('resnet', 15 + (eval_config.increase_time // 7), 96))
    workload.set_infer_workloads(MicrobenchmarkInferWorkload(
        model_list=client_model_list,
        max_request_sec=rps, interval_sec=1, duration=eval_config.increase_time + 65,
    ))
    return workload


def run(system: System, workload: HyperWorkload, server_model_config: list[System.InferModelConfig], tag: str):
    system.launch("memory_perssure", tag, time_stamp=use_time_stamp,
                  infer_model_config=server_model_config)
    workload.launch_workload(system)
    system.stop()
    time.sleep(3)
    system.draw_memory_usage()
    system.draw_trace_cfg()

if run_colsys:
    for num_model in eval_config.num_model_list:
        with mps_thread_percent(eval_config.heavy_mps_infer):
            num_worker = 0
            client_model_list, server_model_config = InferModel.get_multi_model(eval_config.model_list, num_model, num_worker)
            hyper_workload = smooth(rps=eval_config.heavy_rps, client_model_list=client_model_list, infer_only=False)
            system = System(mode=System.ServerMode.ColocateL1, use_sta=True, mps=True, use_xsched=True, has_warmup=True,
                    cuda_memory_pool_gb="13.5", ondemand_adjust=True, train_memory_over_predict_mb=1500,
                    train_mps_thread_percent=eval_config.heavy_mps_train, infer_model_max_idle_ms=5000, port=eval_config.server_port,)
            run(system, hyper_workload, server_model_config, f"colsys-{num_model}")

if run_task_switch:
    for num_model in eval_config.num_model_list:
        num_worker = 0
        client_model_list, server_model_config = InferModel.get_multi_model(eval_config.model_list, num_model, num_worker)
        hyper_workload = smooth(rps=eval_config.heavy_rps, client_model_list=client_model_list, infer_only=False)
        system = System(mode=System.ServerMode.TaskSwitchL1, use_sta=True, mps=False, use_xsched=False, has_warmup=True,
                        cuda_memory_pool_gb="13", train_memory_over_predict_mb=1500, port=eval_config.server_port)
        run(system, hyper_workload, server_model_config, f"task-switch-{num_model}")


if run_um_mps:
    for num_model in eval_config.num_model_list:
        with um_mps(eval_config.heavy_mps_infer):
            num_worker = 1
            client_model_list, server_model_config = InferModel.get_multi_model(eval_config.model_list, num_model, num_worker)
            hyper_workload = smooth(rps=eval_config.heavy_rps, client_model_list=client_model_list, infer_only=False)
            system = System(mode=System.ServerMode.Normal, use_sta=False, mps=True, use_xsched=False, has_warmup=True,
                            train_mps_thread_percent=eval_config.heavy_mps_train, port=eval_config.server_port,
                        max_live_minute=15 + int(eval_config.increase_time / 60 * 15))
            run(system, hyper_workload, server_model_config, f"um-mps-{num_model}")