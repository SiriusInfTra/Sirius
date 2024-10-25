from collections import defaultdict
import glob
import os
import re
import shutil
import sys

path = sys.argv[1]

import shutil
import random
from colorama import init, Fore, Style

def print_aligned(A, B):
    init(autoreset=True)

    def get_console_width():
        try:
            size = shutil.get_terminal_size()
            return size.columns
        except OSError:
            return 80  # 默认宽度

    console_width = get_console_width()

    B_width = 30
    A_width = console_width - B_width

    colors = [Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.WHITE]

    # 选择随机颜色
    color = random.choice(colors)

    output = f"{A:<{A_width}}{B:>{B_width}}"
    print(color + output)



def get_slo(log_file: str) -> float:
    parse_ltc = False
    model_name = None
    ltcs_at_tp = []
    with open(log_file, 'r') as F:
        for line in F.readlines():
            if line.strip() == '':
                parse_ltc = False
                continue
            # if 'InferWorker TRACE' in line:
            mat = re.search(r'\[InferWorker TRACE ([0-9a-zA-Z]+)-(\d+)\]', line)
            if mat is not None:
                parse_ltc = True
                model_name = mat.group(1)
                continue
            if '=' in line or '[ ' in line or ']' in line:
                parse_ltc = False
                model_name = None
                continue

            if parse_ltc: 
                data = line.strip().split(',')
                start_tp = int(data[0].strip())
                end_tp = int(data[1].strip())
                ltc = float(data[-1].strip())
                ltcs_at_tp.append((model_name, start_tp, end_tp, ltc))
                # start_tps.append(start_tp)
                # ltcs.append(ltc)
    std_ltcs = {
        'resnet152': 9.358644,
        'densenet161': 12.586594,
        'efficientnet': 5.287647,
        'efficientvit': 3.7,
        'distilbert_base': 8.030415,
        'distilgpt2': 5.0,
    }
    factor = [1, 2, 3, 4, 5, 6, 7, 8]
    slo = defaultdict(int)
    for model_name, _, _, ltc in ltcs_at_tp:
        if model_name in std_ltcs:
            for f in factor:
                if ltc < std_ltcs[model_name] * f:
                    slo[f] += 1
        else:
            raise ValueError(f'Unknown model {model_name}')
    for k in slo:
        slo[k] /= len(ltcs_at_tp)
    return slo[4]

results = []
for workload_log_path in sorted(list(glob.glob(os.path.join(path, '**', 'workload-log'), recursive=True))):
    workload_log_path = os.path.abspath(workload_log_path)
    work_dir = os.path.dirname(workload_log_path)
    with open(workload_log_path, 'r') as f:
        for line in f:
            if line.startswith('p99 '):
                _, p99, _ = line.split(maxsplit=2)
                break
        else:
            p99 = None
    train_thpt_path = os.path.join(work_dir, 'train_thpt')
    if os.path.exists(train_thpt_path):
        keyword = 'epoch e2e thpt: '
        with open(train_thpt_path, 'r') as f:
            for line in f:
                if keyword in line:
                    t = line.index(keyword)
                    thpt = line[t + len(keyword):].split(' ')[0]
                    break
            else:
                thpt = 'NOT_FOUND'
    else:
        thpt = 'NO_FILE'
    slo = '%.2f' % get_slo(workload_log_path)
    work_dir = os.sep.join(work_dir.split(os.sep)[-2:])
    results.append((work_dir, f"{thpt},,{p99},{slo}"))
def get_value(work_dir):
    work_dir: str = os.path.basename(work_dir)
    work_dir = re.sub(r'-retry-\d+$', '', work_dir)
    variant = 'N'
    if work_dir.endswith('-F') or work_dir.endswith('-I'):
        work_dir, variant = work_dir.rsplit('-', maxsplit=1)
    if 'Normal' not in work_dir and 'Skew' not in work_dir:
        workload, dist = work_dir, 'azure'
    else:
        workload, dist = work_dir.rsplit('-', 1)
    r = 0
    r += {
        'NormalA': 0,
        'NormalB': 1,
        'NormalC': 2,
        'SkewC': 3,
        'azure': 4,
    }[dist] * 100
    r += {
        'static-partition': 0,
        'colsys': 1,
        'task-switch': 2,
        'um-mps': 3,
        'infer-only': 4,
        'infer-only-no-mps': 4,
    }[workload] * 10
    r += {
        'N': 0,
        'F': 1,
        'I': 2,
    }[variant] * 1
    return r
results.sort(key=lambda x: get_value(x[0]))

for w, r in results:
    print(r)
for w, r in results:
    print_aligned(w, r)