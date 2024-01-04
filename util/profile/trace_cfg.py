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
args = parser.parse_args()

trace = args.trace
outdir = args.outdir

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
            trace_list.append(TraceRecord(float(start_point), model_id))

max_time = int(math.ceil(max(
    map(lambda trace_record: trace_record.timepoint, trace_list
))))
num_models = [0 for _ in range(max_time)]
num_types_s = [set() for _ in range(max_time)]
for timepoint, model_id in trace_list:
    num_models[int(timepoint)] += 1
    num_types_s[int(timepoint)].add(model_id)
num_types = [len(s) for s in num_types_s]
timepoints = list(range(max_time))

# print(start_points)

ax = plt.subplot(211)
ax.bar(timepoints, num_models, color='C0', label='rps')
# ax.set_xlabel('time(s)')
ax.set_ylabel('rps')

ax = plt.subplot(212)
ax.bar(timepoints, num_types, color='C1', label='num model')
ax.set_xlabel('time(s)')
ax.set_ylabel('num_model')
# plt.legend()

plt.savefig(os.path.join(outdir, 'trace.svg'))
