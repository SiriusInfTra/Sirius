import os
import shutil
from .config import is_inside_docker, DOCKER_TRITON_MODEL_WKSP

def cp_model(model_list: list[str], n_gpu: int, repo_target_dir: str, models_nbytes: dict[str, int]):
    assert models_nbytes is not None, 'models_nbytes should not be None'
    repo_source_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'server', 'triton_models')
    repo_source_dir = os.path.abspath(repo_source_dir)
    if is_inside_docker():
        triton_model_wksp = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, DOCKER_TRITON_MODEL_WKSP)
        os.system(f'rsync -a {repo_source_dir} {triton_model_wksp}/')
        repo_source_dir = os.path.join(triton_model_wksp, 'triton_models')
        print(f"Prepare model inside docker: Copying triton models from "
              f"{repo_source_dir} to {repo_target_dir}")

    repo_target_dir = os.path.abspath(repo_target_dir)
    device_map = {}
    if os.path.exists(repo_target_dir):
        shutil.rmtree(repo_target_dir)
    os.makedirs(repo_target_dir)
    for i, model_name in enumerate(model_list):
        target_dir = os.path.join(repo_target_dir, model_name)
        os.mkdir(target_dir)
        source_dir = os.path.join(repo_source_dir, model_name.split('-')[0])
        data_name = '1'
        target_data_dir = os.path.join(target_dir, data_name)
        os.mkdir(target_data_dir)
        source_data_dir = os.path.join(source_dir, data_name)
        for file_name in os.listdir(source_data_dir):
            os.link(os.path.join(source_data_dir, file_name), os.path.join(target_data_dir, file_name))
        source_config = os.path.join(source_dir, 'config.pbtxt')
        target_config = os.path.join(target_dir, 'config.pbtxt')
        shutil.copyfile(source_config, target_config)
        with open(target_config, 'r') as f:
            content = f.read()
            content = content.replace(f'name: "{model_name.split("-")[0]}"', f'name: "{model_name}"')
            content = content.replace('gpus: [0]', f'gpus: [{i % n_gpu}]')
            device_map[model_name] = str(i % n_gpu)
        with open(target_config, 'w') as f:
            f.write(content)
    with open(os.path.join(repo_target_dir, 'device_map.txt'), 'w') as f:
        for model_name, device_id in device_map.items():
            f.write(f'{model_name} {device_id}\n')
    with open(os.path.join(repo_target_dir, 'config.conf'), 'w') as f:
        for model_name, nbytes in models_nbytes.items():
            f.write(f'{model_name} = {nbytes}\n')
        

if __name__ == '__main__':
    model_list = []
    for i in range(40):
        model_list.append(f'resnet152-{i}')
    cp_model(model_list, 4, 'triton_models-target')