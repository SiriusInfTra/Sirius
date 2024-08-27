import re, sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from io import StringIO

app = argparse.ArgumentParser("memory trace")
app.add_argument('-l', type=str)
app.add_argument('-b', type=int, default=0)
app.add_argument('-e', type=int, default=3600*1e3)
app.add_argument('-o', type=str, required=True)
app.add_argument('--event', action='store_true')
app.add_argument('--train-all', action=argparse.BooleanOptionalAction, default=True)
args = app.parse_args()
# log = sys.argv[1]
log = args.l
begin_time = args.b
end_time = args.e
outdir = args.o

max_device_num = 32
memory_info_strs = [None] * max_device_num
parse_mem = False
device_id = None
with open(log) as f:
    for line in f.readlines():
        memory_info_match = re.search(r'\[Memory Info \| Device (\d+)\]', line)
        if memory_info_match:
            device_id = int(memory_info_match.group(1))
            memory_info_strs[device_id] = ''
            parse_mem = True
            continue
        if len(line.strip()) == 0:
            parse_mem = False
            device_id = None

        if parse_mem:
            memory_info_strs[device_id] += line

num_devices = len([mem for mem in memory_info_strs if mem is not None])

ax_width = 6
ax_height = 4
hmargin = 0.75

num_subplots = num_devices
if num_devices > 1:
    num_subplots += 1 # plot infer variance

fig, axes = plt.subplots(num_subplots, 1, 
                         figsize=(ax_width, ax_height*num_devices + hmargin*(num_subplots-1)))

print(len(axes))

infer_mem_all_devices = []

for i in range(num_devices):
    ax = axes[i]
    df = pd.read_csv(StringIO(memory_info_strs[i]), sep=',', header=0, index_col=0)

    memory_transform = lambda x: float(re.search(r'(\d+\.\d+) Mb', x).group(1))

    df.columns = df.columns.str.strip()
    # print(df.columns)
    timeline = df.index - df.index[0]
    total_mem = df['TotalMem'].apply(memory_transform)
    train_all_mem = df['TrainAllMem'].apply(memory_transform)
    infer_mem = df['InferMem'].apply(memory_transform)

    total_mem = max(total_mem)

    if args.train_all:
        train_mem = train_all_mem

    infer_mem, train_mem = map(np.array, [infer_mem, train_mem])
    infer_mem_all_devices.append(infer_mem)
    # invert_train_mem = total_mem - train_mem
    infer_train_mem = infer_mem + train_mem

    ax.plot(timeline, infer_mem, label='Infer', linewidth=0.2)
    ax.fill_between(timeline, 0, infer_mem, alpha=0.3)

    ax.plot(timeline, infer_train_mem, label='Train', linewidth=0.2)
    ax.fill_between(timeline, infer_mem, infer_train_mem, alpha=0.3)

    ax.set_ylim(0, total_mem)
    ax.set_xlim(0, min(max(timeline), end_time - begin_time))

    ax.set_xlabel("Time (ms)", fontsize=14)
    ax.set_ylabel("GPU Memory (MB)",fontsize=14)
    ax.legend()

# memory info are expected to have same timestamp index
infer_mem_all_devices = np.array(infer_mem_all_devices)
infer_mem_all_devices = infer_mem_all_devices.transpose()
if num_devices > 1:
    ax = axes[-1]
    min_infer_mem = np.min(infer_mem_all_devices, axis=1)
    max_infer_mem = np.max(infer_mem_all_devices, axis=1)
    delta_infer_mem = max_infer_mem - min_infer_mem
    ax.plot(timeline, delta_infer_mem, label='Infer variance', linewidth=0.5)
    ax.set_xlabel("Time (ms)", fontsize=14)
    ax.set_ylabel("Infer Memory |Max-Min| (MB)",fontsize=14)

# plt.legend()

plt.savefig(f'{outdir}/memory-trace.svg')
