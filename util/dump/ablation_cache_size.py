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
            return 80  # default width

    console_width = get_console_width()

    B_width = 30
    A_width = console_width - B_width

    colors = [Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.WHITE]

    # select random color
    color = random.choice(colors)

    output = f"{A:<{A_width}}{B:>{B_width}}"
    print(color + output)


result = []
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
    waste = None
    thpt = None
    if os.path.exists(train_thpt_path):
        with open(train_thpt_path, 'r') as f:
            for line in f:
                keyword = 'cancel waste '
                if keyword in line:
                    t = line.index(keyword)
                    if waste is None:
                        waste = line[t + len(keyword):].split(' ')[1].removeprefix('(').removesuffix(')').removesuffix('%')
                keyword = 'epoch e2e thpt: '
                if keyword in line:
                    t = line.index(keyword)
                    if thpt is None:
                        thpt = line[t + len(keyword):].split(' ')[0]
    assert waste is not None
    assert thpt is not None
    
    profile_log_path = os.path.join(work_dir, 'profile-log.log')
    load_avg = None
    with open(profile_log_path, 'r') as f:
        for line in f:
            if line.startswith('InferLoadParam: '):
                keyword = 'avg '
                _, load_avg, _ = line[line.index(keyword):].split(' ', maxsplit=2)
    assert load_avg is not None
    cache_size = os.path.basename(work_dir).split('-')[1].removesuffix('GB')
    work_dir = os.sep.join(work_dir.split(os.sep)[-2:])
    result.append((cache_size, p99, load_avg, thpt, waste))

result = sorted(result, key=lambda x: float(x[0]))
for r in result:
    print('\t\t'.join(r))
