from runner import *

set_global_seed(42)

use_time_stamp = True

@dataclass
class AzureConfig:
    model_list = [InferModel.InceptionV3, InferModel.ResNet152, InferModel.DenseNet161, InferModel.DistilBertBase]
    heavy_rps = 100
    heavy_num_model = 64
    heavy_mps_infer = 70
    heavy_mps_train = 30
    
    light_rps = 10
    light_num_model = 32
    light_mps_infer = 30
    light_mps_train = 70
    
    server_port = "18480"
    
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



def run(system: System, workload: HyperWorkload, server_model_config, tag: str):
    
    system.launch("dynamic-overhead", tag, time_stamp=use_time_stamp,
                  infer_model_config=server_model_config)
    workload.launch_workload(system)
    system.stop()
    time.sleep(5)
    system.draw_memory_usage()
    system.draw_trace_cfg()


# for max_idle_ms in [3000]:
# for max_idle_ms in [10000, 1000*1000]:
for max_idle_ms in [500, 1000, 2000, 4000, 5000, 8000, 10000, 1000*1000]:
    num_worker = 0
    with mps_thread_percent(60):
        client_model_list, server_model_config = InferModel.get_multi_model(eval_config.model_list, eval_config.heavy_num_model, num_worker)
        hyper_workload = azure(rps=eval_config.heavy_rps, client_model_list=client_model_list, infer_only=True)
        system = System(mode=System.ServerMode.ColocateL1, use_sta=True, mps=True, use_xsched=True, 
                cuda_memory_pool_gb="13.5", ondemand_adjust=True, train_memory_over_predict_mb=1500,
                train_mps_thread_percent=40, memory_pool_policy=System.MemoryPoolPolicy.NextFit,
                infer_model_max_idle_ms=max_idle_ms, has_warmup=True)
        tag = f"{max_idle_ms}" if max_idle_ms < 1000*1000 else "inf"
        run(system, hyper_workload, server_model_config, tag)