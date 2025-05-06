from __future__ import annotations
import sys
import os
import re

def replace_nan(s: str) -> str:
    if s == 'nan':
        return '-1'
    else:
        return s

def get_stats(path: str) -> tuple[str, str, str, str, str]:
    avg_batch_size = .0
    e2e_thpt = .0
    p99 = .0
    hit_rato = .0
    ret_set = []
    if os.path.exists(os.path.join(path, 'workload-log')):
        with open(os.path.join(path, 'workload-log'), 'r') as f:
            for line in f:
                if 'p99' in line:
                    arr = line.split()
                    p99 = arr[arr.index('p99') + 1]
                    p99 = replace_nan(p99)
                    ret_set.append(line)
                    break
    else:
        return os.path.basename(path), e2e_thpt, avg_batch_size, p99, hit_rato
    if os.path.exists(os.path.join(path, 'train_thpt')):
        with open(os.path.join(path, 'train_thpt'), 'r') as f:
            for line in f:
                if 'avg_batch_size' in line and 'global batch thpt' not in line:
                    arr = line.split()
                    avg_batch_size = arr[arr.index('avg_batch_size') + 1].replace(',', '')
                    avg_batch_size = replace_nan(avg_batch_size)
                    ret_set.append(line)
                if 'epoch e2e thpt: ' in line:
                    arr = line.split()
                    e2e_thpt = arr[arr.index('thpt:') + 1]
                    e2e_thpt = replace_nan(e2e_thpt)
                    ret_set.append(line)
    if os.path.exists(os.path.join(path, 'profile-log.log')):
        with open(os.path.join(path, 'profile-log.log')) as f:
            for line in f:
                if 'InferModelColdCacheHit' in line and 'no record' not in line:
                    arr = line.split()
                    hit_rato = float(arr[arr.index('avg') + 1])
                    ret_set.append(line)
    return os.path.basename(path), e2e_thpt, avg_batch_size, p99, hit_rato


NAME_ORDER = [
    'UNKNOWN',
    'static-partition-F',
    'static-partition-I',
    'sirius',
    'task-switch',
    'um-mps',
    'infer-only',
]
def get_index(name: str) -> int:
    index = 0
    if '-high' in name:
        index += 1 * len(NAME_ORDER)
        name = name.replace('-high', '')
    elif '-low' in name:
        index += 0 * len(NAME_ORDER)
        name = name.replace('-low', '')
    name = re.sub(r'-retry-\d+', '', name)
    try:
        index += NAME_ORDER.index(name)
    except ValueError:
        index += NAME_ORDER.index('UNKNOWN')
    return index


    
base_dir_name = sys.argv[1]
ret = []
if os.path.exists(os.path.join(os.path.join(base_dir_name, 'workload-log'))):
    ret.append(get_stats(base_dir_name))
else:
    for dir_name in os.listdir(base_dir_name):
        ret.append(get_stats(os.path.join(base_dir_name, dir_name)))
ret.sort(key=lambda x: get_index(x[0]))
for name, e2e_thpt, avg_batch_size, p99, hit_rato in ret:
    print(name, e2e_thpt, avg_batch_size, p99, hit_rato)
            