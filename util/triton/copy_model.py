import os
import shutil

replicate_n = 40

work_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'server', 'triton_models')
work_dir = os.path.abspath(work_dir)
for model_name in os.listdir(work_dir):
    model_dir = os.path.join(work_dir, model_name)
    if ('-' in model_name) or (not os.path.isdir(model_dir)):
        continue
    for n in range(1, replicate_n):
        new_model_dir = os.path.join(model_dir, os.pardir, f'{model_name}-{n}')
        new_model_dir = os.path.abspath(new_model_dir)
        if os.path.exists(new_model_dir):
            print(f"Model directory {new_model_dir} already exists, skipping.")
            continue
        os.mkdir(new_model_dir)
        data_name = '1'
        model_data_dir = os.path.join(model_dir, data_name)
        new_model_data_dir = os.path.join(new_model_dir, data_name)
        os.mkdir(new_model_data_dir)
        for file_name in os.listdir(model_data_dir):
            os.link(os.path.join(model_data_dir, file_name), os.path.join(new_model_data_dir, file_name))
        shutil.copyfile(os.path.join(model_dir, 'config.pbtxt'), os.path.join(new_model_dir, 'config.pbtxt'))
        config_path = os.path.join(new_model_dir, 'config.pbtxt')
        with open(config_path, 'r') as file:
            config_data = file.read()
        new_config_data = config_data.replace(f'name: "{model_name}"', f'name: "{model_name}-{n}"')
        # Write back the modified content
        with open(config_path, 'w') as file:
            file.write(new_config_data)