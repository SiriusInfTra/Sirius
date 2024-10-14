import os
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser(description='Copy and modify model configurations.')
    parser.add_argument('--model_name', type=str, required=True, help='The name of the model to copy and modify.')
    args = parser.parse_args()

    # 原始模型目录
    original_model_dir = args.model_name

    # 目标根目录
    target_root_dir = '.'

    # 复制8份
    for i in range(1, 10):
        # 新的模型目录名
        new_model_dir = os.path.join(target_root_dir, f'{args.model_name}-{i}')
        if os.path.exists(new_model_dir):
            print(f"模型目录 {new_model_dir} 已经存在，跳过。")
            continue
        # 复制目录
        shutil.copytree(original_model_dir, new_model_dir)
        
        # 修改config.pbtxt
        config_path = os.path.join(new_model_dir, 'config.pbtxt')
        with open(config_path, 'r') as file:
            config_data = file.read()
        
        # 修改name字段
        new_config_data = config_data.replace(f'name: "{args.model_name}"', f'name: "{args.model_name}-{i}"')
        
        # 写回修改后的内容
        with open(config_path, 'w') as file:
            file.write(new_config_data)

    print("模型复制和配置文件修改完成。")

if __name__ == '__main__':
    main()
