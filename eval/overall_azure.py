from runner import *
from dataclasses import dataclass

set_global_seed(42)

use_time_stamp = True

run_pct_mps = False
run_colsys = True
run_um_mps = False
run_task_switch = False
run_infer_only = True

@dataclass
class AzureConfig:
    model_list = [InferModel.InceptionV3, InferModel.ResNet152, InferModel.DenseNet161, InferModel.DistilBertBase]
    heavy_rps = 100
    heavy_num_model = 64
    heavy_mps_infer = 60
    heavy_mps_train = 40
    
    light_rps = 10
    light_num_model = 32
    light_mps_infer = 30
    light_mps_train = 70
    
    server_port = "18380"
    
eval_config = AzureConfig()


def azure(rps, client_model_list, infer_only=True):
    workload = HyperWorkload(concurrency=2048, duration=140, delay_before_infer=30,
                            warmup=5, delay_after_warmup=5, delay_before_profile=5)
    InferModel.reset_model_cnt()
    if not infer_only:
        workload.set_train_workload(train_workload=TrainWorkload('resnet', 15, 96))
    workload.set_infer_workloads(AzureInferWorkload(
        AzureInferWorkload.TRACE_D01,
        model_list=client_model_list,
        max_request_sec=rps, interval_sec=1, period_num=65, func_num=16 * 1000, 
    ))
    
    return workload




def run(system: System, workload: HyperWorkload, server_model_config: list[System.InferModelConfig], tag: str):
    system.launch("try-infer-mps-azure", tag, time_stamp=use_time_stamp,
                  infer_model_config=server_model_config)
    workload.launch_workload(system)
    system.stop()
    time.sleep(3)
    system.draw_memory_usage()
    system.draw_trace_cfg()


if run_pct_mps:
    num_worker = 1
    # infer only heavy
    for mps_pct in [0]:
        with mps_thread_percent(mps_pct):
            client_model_list, server_model_config = InferModel.get_multi_model(eval_config.model_list, eval_config.heavy_num_model, num_worker)
            hyper_workload = azure(rps=eval_config.heavy_rps, client_model_list=client_model_list, infer_only=True)
            system = System(mode=System.ServerMode.Normal, use_sta=False, mps=mps_pct > 0, use_xsched=False, port=eval_config.server_port)
            run(system, hyper_workload, server_model_config, f"heavy-{mps_pct}")
    
    # infer only light
    for mps_pct in [0]:
        with mps_thread_percent(mps_pct):
            client_model_list, server_model_config = InferModel.get_multi_model(eval_config.model_list, eval_config.light_num_model, num_worker)
            hyper_workload = azure(rps=eval_config.light_rps, client_model_list=client_model_list, infer_only=True)
            system = System(mode=System.ServerMode.Normal, use_sta=False, mps=mps_pct > 0, use_xsched=False, port=eval_config.server_port)
            run(system, hyper_workload, server_model_config, f"light-{mps_pct}")







def run(system: System, workload: HyperWorkload, server_model_config: str, tag: str):
    system.launch("overall-azure", tag, time_stamp=use_time_stamp,
                  infer_model_config=server_model_config)
    workload.launch_workload(system)
    system.stop()
    time.sleep(5)
    system.draw_memory_usage()
    system.draw_trace_cfg()


if run_colsys:
    num_worker = 0
    # colsys heavy
    with mps_thread_percent(eval_config.heavy_mps_infer):
        client_model_list, server_model_config = InferModel.get_multi_model(eval_config.model_list, eval_config.heavy_num_model, num_worker)
        hyper_workload = azure(rps=eval_config.heavy_rps, client_model_list=client_model_list, infer_only=False)
        system = System(mode=System.ServerMode.ColocateL1, use_sta=True, mps=True, use_xsched=True, port=eval_config.server_port, 
                        cuda_memory_pool_gb="13.5", ondemand_adjust=True, train_memory_over_predict_mb=1500,
                        train_mps_thread_percent=eval_config.heavy_mps_train, infer_model_max_idle_ms=4000)
        run(system, hyper_workload, server_model_config, "colsys-heavy")

    # colsys light
    with mps_thread_percent(eval_config.light_mps_infer):
        client_model_list, server_model_config = InferModel.get_multi_model(eval_config.model_list, eval_config.light_num_model, num_worker)
        hyper_workload = azure(rps=eval_config.light_rps, client_model_list=client_model_list, infer_only=False)

        system = System(mode=System.ServerMode.ColocateL1, use_sta=True, mps=True, use_xsched=True, port=eval_config.server_port, 
                        cuda_memory_pool_gb="13.5", ondemand_adjust=True, train_memory_over_predict_mb=1500,
                        train_mps_thread_percent=eval_config.light_mps_train, infer_model_max_idle_ms=4000)
        run(system, hyper_workload, server_model_config, "colsys-light")


if run_um_mps:
    num_worker = 1
    # um+mps heavy
    with um_mps(eval_config.heavy_mps_infer):
        client_model_list, server_model_config = InferModel.get_multi_model(eval_config.model_list, eval_config.heavy_num_model, num_worker)
        hyper_workload = azure(rps=eval_config.heavy_rps, client_model_list=client_model_list, infer_only=False)

        system = System(mode=System.ServerMode.Normal, use_sta=False, mps=True, use_xsched=False, port=eval_config.server_port,
                        train_mps_thread_percent=eval_config.heavy_mps_train)
        run(system, hyper_workload, server_model_config, "um-mps-heavy")
    # um+mps light
    with um_mps(eval_config.light_mps_infer):
        client_model_list, server_model_config = InferModel.get_multi_model(eval_config.model_list, eval_config.light_num_model, num_worker)
        hyper_workload = azure(rps=eval_config.light_rps, client_model_list=client_model_list, infer_only=False)

        system = System(mode=System.ServerMode.Normal, use_sta=False, mps=True, use_xsched=False, port=eval_config.server_port,
                        train_mps_thread_percent=eval_config.light_mps_train)
        run(system, hyper_workload, server_model_config, "um-mps-light")


if run_task_switch:
    num_worker = 0
    # task switch heavy
    client_model_list, server_model_config = InferModel.get_multi_model(eval_config.model_list, eval_config.heavy_num_model, num_worker)
    hyper_workload = azure(rps=eval_config.heavy_rps, client_model_list=client_model_list, infer_only=False)
    system = System(mode=System.ServerMode.TaskSwitchL1, use_sta=True, mps=False, use_xsched=False, port=eval_config.server_port,
                    cuda_memory_pool_gb="13", train_memory_over_predict_mb=1500)
    run(system, hyper_workload, server_model_config, "task-switch-heavy")
    # task switch light
    client_model_list, server_model_config = InferModel.get_multi_model(eval_config.model_list, eval_config.light_num_model, num_worker)
    hyper_workload = azure(rps=eval_config.light_rps, client_model_list=client_model_list, infer_only=False)
    system = System(mode=System.ServerMode.TaskSwitchL1, use_sta=True, mps=False, use_xsched=False, port=eval_config.server_port,
                    cuda_memory_pool_gb="13", train_memory_over_predict_mb=1500)
    run(system, hyper_workload, server_model_config, "task-switch-light")


if run_infer_only:
    num_worker = 1
    # infer only heavy
    with mps_thread_percent(eval_config.heavy_mps_infer):
        client_model_list, server_model_config = InferModel.get_multi_model(eval_config.model_list, eval_config.heavy_num_model, num_worker)
        hyper_workload = azure(rps=eval_config.heavy_rps, client_model_list=client_model_list, infer_only=True)
        system = System(mode=System.ServerMode.Normal, use_sta=False, mps=True, use_xsched=False, port=eval_config.server_port)
        run(system, hyper_workload, server_model_config, "infer-only-heavy")
    with mps_thread_percent(eval_config.light_mps_infer):
        # infer only light
        client_model_list, server_model_config = InferModel.get_multi_model(eval_config.model_list, eval_config.heavy_num_model, num_worker)
        hyper_workload = azure(rps=eval_config.light_rps, client_model_list=client_model_list, infer_only=True)
        system = System(mode=System.ServerMode.Normal, use_sta=False, mps=True, use_xsched=False, port=eval_config.server_port)
        run(system, hyper_workload, server_model_config, "infer-only-light")




