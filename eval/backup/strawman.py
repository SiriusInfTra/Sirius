from runner import *
from dataclasses import dataclass

set_global_seed(42)

use_time_stamp = True

run_strawman = True
run_pct_mps = True

@dataclass
class StrawmanConfig:
    model_list = [InferModel.ResNet152]
    heavy_rps = 100
    heavy_num_model = 36
    heavy_mps_infer = 60
    heavy_mps_train = 40
    # heavy_try_mps_pct_list = [0]
    heavy_try_mps_pct_list = [0, heavy_mps_infer]

    server_port = "18780"
    increase_time = 60
    
eval_config = StrawmanConfig()

def smooth(rps, client_model_list, infer_only=True):
    workload = HyperWorkload(concurrency=2048, duration=eval_config.increase_time + 140, warmup=5,
                             wait_warmup_done_sec=5, wait_train_setup_sec=30,
                             wait_stable_before_start_profiling_sec=10)
    InferModel.reset_model_cnt()
    if not infer_only:
        workload.set_train_workload(train_workload=TrainWorkload('resnet', 15 + (eval_config.increase_time // 7), 96))
    workload.set_infer_workloads(MicrobenchmarkInferWorkload(
        model_list=client_model_list,
        max_request_sec=rps, interval_sec=1, duration=eval_config.increase_time + 65,
    ))
    return workload



def run(system: System, workload: HyperWorkload, server_model_config: str, tag: str):
    system.launch("strawman-smooth", tag, time_stamp=use_time_stamp,
                  infer_model_config=server_model_config)
    workload.launch_workload(system)
    system.stop()
    time.sleep(5)
    system.draw_memory_usage()
    system.draw_trace_cfg()


if run_pct_mps:
    num_worker = 1
    # infer only heavy
    client_model_list, server_model_config = InferModel.get_multi_model(eval_config.model_list, eval_config.heavy_num_model, num_worker)
    for mps_pct in eval_config.heavy_try_mps_pct_list:
        with mps_thread_percent(mps_pct):
            client_model_list, server_model_config = InferModel.get_multi_model(eval_config.model_list, eval_config.heavy_num_model, num_worker)
            hyper_workload = smooth(rps=eval_config.heavy_rps, client_model_list=client_model_list, infer_only=True)
            system = System(mode=System.ServerMode.Normal, use_sta=False, mps=mps_pct > 0, use_xsched=False, port=eval_config.server_port)
            run(system, hyper_workload, server_model_config, f"heavy-{mps_pct}")
    
    # infer only light
    for mps_pct in eval_config.light_try_mps_pct_list:
        with mps_thread_percent(mps_pct):
            client_model_list, server_model_config = InferModel.get_multi_model(eval_config.model_list, eval_config.light_num_model, num_worker)
            hyper_workload = smooth(rps=eval_config.light_rps, client_model_list=client_model_list, infer_only=True)
            system = System(mode=System.ServerMode.Normal, use_sta=False, mps=mps_pct > 0, use_xsched=False, port=eval_config.server_port)
            run(system, hyper_workload, server_model_config, f"light-{mps_pct}")


if run_strawman:
    num_worker = 0
    with mps_thread_percent(eval_config.heavy_mps_infer):
        client_model_list, server_model_config = InferModel.get_multi_model(eval_config.model_list, eval_config.heavy_num_model, num_worker)
        hyper_workload = smooth(rps=eval_config.heavy_rps, client_model_list=client_model_list, infer_only=False)
        system = System(mode=System.ServerMode.ColocateL2, use_sta=False, mps=True, use_xsched=False, port=eval_config.server_port,
                        ondemand_adjust=True, train_memory_over_predict_mb=1500,
                        has_warmup=True, infer_model_max_idle_ms=5000,
                        train_mps_thread_percent=eval_config.heavy_mps_train)
        run(system, hyper_workload, server_model_config, "strawman-heavy")
