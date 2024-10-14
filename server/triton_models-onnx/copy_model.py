import os
import shutil

# 原始模型目录
original_model_dir = 'resnet152'

# 目标根目录
target_root_dir = '.'

# 复制8份
for i in range(1, 8):
    # 新的模型目录名
    new_model_dir = os.path.join(target_root_dir, f'resnet152-{i}')
    
    # 复制目录
    shutil.copytree(original_model_dir, new_model_dir)
    
    # 修改config.pbtxt
    config_path = os.path.join(new_model_dir, 'config.pbtxt')
    with open(config_path, 'r') as file:
        config_data = file.read()
    
    # 修改name字段
    new_config_data = config_data.replace('name: "resnet152"', f'name: "resnet152-{i}"')
    
    # 写回修改后的内容
    with open(config_path, 'w') as file:
        file.write(new_config_data)

print("模型复制和配置文件修改完成。")
