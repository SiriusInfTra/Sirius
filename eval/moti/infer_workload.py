import sys
sys.path.append("./eval/")

from runner import *

set_global_seed(42)
use_time_stamp = True


def get_unique_port():
    cuda_device = os.environ['CUDA_VISIBLE_DEVICES']
    assert ',' not in cuda_device
    try:
        cuda_device = int(cuda_device)
    except:
        cuda_device = GPU_UUIDs.index(cuda_device)
    port = 18100
    port += cuda_device
    return port


class AzureConfig:
    model_list = [InferModel.DenseNet161, InferModel.EfficientNetV2_s, 
                  InferModel.EfficientViT_b2, InferModel.DistilBertBase, 
                  InferModel.ResNet152, InferModel.DistilGPT2]
    rps = 150
    num_model = 64
    server_port = str(get_unique_port())

    interval_sec = 60
    period_num = 180


def azure(rps, client_model_list):
    workload = HyperWorkload(concurrency=2048, 
                             warmup=5, 
                             wait_warmup_done_sec=5,
                             wait_stable_before_start_profiling_sec=0)
    InferModel.reset_model_cnt()
    workload.set_infer_workloads(AzureInferWorkload(
        AzureInferWorkload.TRACE_D01,
        model_list=client_model_list,
        max_request_sec=150, 
        interval_sec=AzureConfig.interval_sec, 
        period_num=AzureConfig.period_num, 
        func_num=AzureConfig.num_model * 3, # 3 is a suitable number
        sort_trace_by='var_v2'
    ))
    # workload.set_infer_workloads(MicrobenchmarkInferWorkload(
    #     model_list=client_model_list,
    #     max_request_sec=rps, 
    #     interval_sec=AzureConfig.interval_sec, 
    #     period_num=AzureConfig.period_num,
    # ))
    return workload


def run(system: System, workload: HyperWorkload, 
        server_model_config: str, tag: str):
    system.launch("infer-workload", tag, 
                time_stamp=use_time_stamp,
                infer_model_config=server_model_config,
                dcgmi=True, 
                fake_launch=False)
    workload.launch_workload(system, fake_launch=False)
    system.stop()
    time.sleep(5)
    system.draw_memory_usage()
    system.draw_trace_cfg(time_scale=60)


# AzureConfig.interval_sec = 60

max_live_minute=int(AzureConfig.interval_sec * AzureConfig.period_num / 60 + 10)

client_model_list, server_model_config = \
    InferModel.get_multi_model(AzureConfig.model_list, AzureConfig.num_model, 1)
hyper_workload = azure(
    rps=AzureConfig.rps,
    client_model_list=client_model_list,
)
system = System(
    mode=System.ServerMode.Normal,
    use_sta=False, mps=False, use_xsched=False,
    port=AzureConfig.server_port,
    has_warmup=True,
    max_live_minute=max_live_minute,
)

# infer w/ less memory
# system = System(
#     mode=System.ServerMode.Normal,
#     use_sta=True, mps=False, use_xsched=False,
#     port=AzureConfig.server_port,
#     has_warmup=True,
#     max_live_minute=max_live_minute,
#     max_cache_nbytes=int(4.5 * 1024 * 1024 * 1024),
#     cuda_memory_pool_gb=6
# )

run(system, hyper_workload, server_model_config, 
    f"azure-interval-{AzureConfig.interval_sec}-period-{AzureConfig.period_num}")




