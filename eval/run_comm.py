import os
import argparse
from runner import *
from dataclasses import dataclass

use_time_stamp = True
retry_if_fail = False
skip_fail = False

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


@dataclass
class HighLoad:
    rps: int = 50
    mps_infer: int = 35
    mps_train: int = 60
    enable: bool = True

@dataclass
class LowLoad:
    rps: int = 5
    mps_infer: int = 30
    mps_train: int = 70
    enable: bool = True

@dataclass
class HybridLoad:
    high_rps = HighLoad.rps
    low_rps = LowLoad.rps
    mps_infer: int = HighLoad.rps
    mps_train: int = HighLoad.rps
    enable: bool = True


def _run(system: System, workload: HyperWorkload, server_model_config: str, unit: str, tag: str):
    try:
        system.launch(unit, tag, time_stamp=use_time_stamp,
                    infer_model_config=server_model_config, fake_launch=False)
        workload.launch_workload(system, fake_launch=False)
        system.stop()
    except Exception as e:
        print(f"Failed to run {unit} {tag}: {e}")
        if retry_if_fail:
            print(f"\n\x1b[33;1m### Retry [{unit} {tag}] ###\x1b[0m")
            system.launch(unit, f'{tag}-retry', time_stamp=use_time_stamp,
                    infer_model_config=server_model_config)
            workload.launch_workload(system)
            system.stop()
        else:
            raise e
    time.sleep(5)
    system.draw_memory_usage()
    system.draw_trace_cfg()
    system.calcuate_train_thpt()
    

def run(system: System, workload: HyperWorkload, server_model_config: str, unit: str, tag: str):
    if skip_fail:
        try:
            _run(system, workload, server_model_config, unit, tag)
        except Exception as e:
            print(f"\n\x1b[33;1m### Skip {unit} {tag}: {e} ###\x1b[0m")
    else:
        _run(system, workload, server_model_config, unit, tag)


