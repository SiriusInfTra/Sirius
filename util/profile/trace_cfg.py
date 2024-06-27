import matplotlib.pyplot as plt
import argparse
import os
import math
from typing import NamedTuple

class TraceRecord(NamedTuple):
    timepoint: float
    model_id: str

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--trace', type=str, required=True)
parser.add_argument('-o', '--outdir', type=str, required=True)
parser.add_argument('--time-scale', type=float, default=1)
args = parser.parse_args()

trace = args.trace
outdir = args.outdir
time_scale = args.time_scale

trace_list = [] # (start_point, model_id)
with open(trace, 'r') as f:
    lines = f.readlines()
    start = False
    for l in lines:
        if not start:
            if l.strip() != '# start_point,model_id':
                continue
            else:
                start = True
        else:
            start_point, model_id = l.strip().split(',')
            trace_list.append(TraceRecord(float(start_point) / time_scale, model_id))

max_time = int(math.ceil(max(
    map(lambda trace_record: trace_record.timepoint, trace_list
))))
num_models = [0 for _ in range(max_time)]
num_types_s = [set() for _ in range(max_time)]
for timepoint, model_id in trace_list:
    num_models[int(timepoint)] += 1 / time_scale
    num_types_s[int(timepoint)].add(model_id)
num_types = [len(s) for s in num_types_s]
timepoints = list(range(max_time))

# print(start_points)

fig, axes = plt.subplots(2, 2)
fig.subplots_adjust(hspace=0.3, wspace=0.3)

# RPS
axes[0][0].bar(timepoints, num_models, color='C0', label='rps')
axes[0][0].set_ylabel('rps')
axes[0][0].set_xlabel('time(s)')

# RPS CDF
axes[0][1].ecdf(num_models, label='RPS CDF')
axes[0][1].set_xlabel('RPS')
axes[0][1].set_ylabel('CDF')

# num_models
axes[1][0].bar(timepoints, num_types, color='C1', label='num model')
axes[1][0].set_xlabel('time(s)')
axes[1][0].set_ylabel('num_model')

# num_models CDF
axes[1][1].ecdf(num_types, label='#model CDF')
axes[1][1].set_xlabel('#model')
axes[1][1].set_ylabel('CDF')

plt.savefig(os.path.join(outdir, 'trace.svg'))
