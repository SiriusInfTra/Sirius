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

memory_info_str = ""
parse_mem = False
with open(log) as f:
    for line in f.readlines():
        if '[Memory Info]' in line:
            parse_mem = True
            continue
        if len(line.strip()) == 0:
            parse_mem = False

        if parse_mem:
            memory_info_str += line

df = pd.read_csv(StringIO(memory_info_str), sep=',', header=0, index_col=0)


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

fig, ax = plt.subplots()
infer_mem, train_mem = map(np.array, [infer_mem, train_mem])
# invert_train_mem = total_mem - train_mem
infer_train_mem = infer_mem + train_mem


ax.plot(timeline, infer_mem, label='Infer', linewidth=0.2)
ax.fill_between(timeline, 0, infer_mem, alpha=0.3)

ax.plot(timeline, infer_train_mem, label='Train', linewidth=0.2)
ax.fill_between(timeline, infer_mem, infer_train_mem, alpha=0.3)

# if args.event:
#     ax.vlines([t for t, e in events], 0.0, total_mem, color='black', linewidth=0.5, linestyles='dashed')


# print([t for t, e in events])
# ax.vlines([t for t, e in events], 0, total_mem, 
#           color='black', linewidth=0.5, linestyles='dashed')


ax.set_ylim(0, total_mem)
ax.set_xlim(0,
            min(max(timeline), end_time - begin_time))

ax.set_xlabel("Time (ms)", fontsize=14)
ax.set_ylabel("GPU Memory (MB)",fontsize=14)
plt.legend()
plt.savefig(f'{outdir}/memory-trace.svg')
