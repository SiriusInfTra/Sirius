import os
import glob
import shutil

work_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'server', 'triton_models')
work_dir = os.path.abspath(work_dir)
for model_dir in glob.glob(os.path.join(work_dir, '*-*')):
    print(model_dir)
    shutil.rmtree(model_dir)