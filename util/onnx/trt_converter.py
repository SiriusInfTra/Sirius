from __future__ import annotations
from enum import Enum
import os
import subprocess


class TrtModelConfig(Enum):
    Vison = {
        'input0': '3x224x224',
    }
    DistilGPT2 = {
        'input_ids': '64',
    }
    DistilBert = {
        'input_ids': '64',
        'attention_mask': '64',
    }

def convert_to_tensorrt(model_name: str, batch_size: int, model_config: TrtModelConfig) -> bool:
    if 'COLSYS_TENSORRT_HOME' not in os.environ:
        return False
    trt_model_path = f"server/triton_models/{model_name}/1/model.plan"
    trt_config_path = f"server/triton_models/{model_name}/config.pbtxt"
    config_template = """
name: "{model_name}"
platform: "tensorrt_plan"
max_batch_size: {max_batch_size}
instance_group [ 
  { 
    count: 1 
    kind: KIND_GPU
    gpus: [0] 
  }
]
""".strip().replace("{model_name}", model_name).replace("{max_batch_size}", str(batch_size))
    if not os.path.exists(os.path.dirname(trt_model_path)):
        os.makedirs(os.path.dirname(trt_model_path))
    if not os.path.exists(os.path.dirname(trt_config_path)):
        os.makedirs(os.path.dirname(trt_config_path))
    tensorrt_home = os.environ['COLSYS_TENSORRT_HOME']
    amd64_home = f'{tensorrt_home}/targets/x86_64-linux-gnu'
    envs = os.environ.copy()
    if 'LD_LIBRARY_PATH' in envs:
        envs['LD_LIBRARY_PATH'] = f'{amd64_home}/lib:{envs["LD_LIBRARY_PATH"]}'
    else:
        envs['LD_LIBRARY_PATH'] = f'{amd64_home}/lib'
    model_config_dict = model_config.value
    args = [
        f'{amd64_home}/bin/trtexec', 
        f'--onnx={model_name}.onnx', 
        f'--saveEngine={trt_model_path}'
    ]
    if len(model_config_dict) > 0:
        min_shapes = ','.join([f'{input_name}:1x{shape}' for input_name, shape in model_config_dict.items()])
        opt_shapes = ','.join([f'{input_name}:1x{shape}' for input_name, shape in model_config_dict.items()])
        max_shapes = ','.join([f'{input_name}:{batch_size}x{shape}' for input_name, shape in model_config_dict.items()])
        shapes = ','.join([f'{input_name}:1x{shape}' for input_name, shape in model_config_dict.items()])
        args.extend([
            f'--minShapes={min_shapes}',
            f'--optShapes={opt_shapes}',
            f'--maxShapes={max_shapes}',
            f'--shapes={shapes}'
        ])
    subprocess.run(args, env=envs)
    # print(f'Exec args = {args}, envs = {envs}.')
    with open(trt_config_path, "w") as f:
        f.write(config_template)
    return True