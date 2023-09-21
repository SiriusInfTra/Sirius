import re, sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

app = argparse.ArgumentParser("memory trace")
app.add_argument('-l', type=str)
app.add_argument('-b', type=int, default=0)
app.add_argument('-e', type=int, default=3600*1e3)
app.add_argument('--event', action='store_true')
args = app.parse_args()
# log = sys.argv[1]
log = args.l
begin_time = args.b
end_time = args.e

timeline = []
infer_mem, train_mem, total_mem = [], [], []
events = []
with open(log) as f:
    parse_mem = False
    paser_ev = False
    for line in f.readlines():
        if '[Memory Info]' in line:
            parse_mem = True
            continue
        if '[Event Info]' in line:
            paser_ev = True
            continue
        if len(line.strip()) == 0:
            parse_mem = False
            paser_ev = False

        if parse_mem:
            m = re.search(r'(\d+(\.\d+)?):', line)
            timeline.append(float(m.group(1)) - begin_time)

            m = re.search(r'Infer (\d+\.\d+) Mb', line)
            infer_mem.append(float(m.group(1)))

            m = re.search(r'Train (\d+\.\d+) Mb', line)
            train_mem.append(float(m.group(1)))

            m = re.search(r'Total (\d+\.\d+) Mb', line)
            total_mem.append(float(m.group(1)))

        if paser_ev:
            m = re.search(r'(\d+(\.\d+)?):', line)
            t = float(m.group(1)) - begin_time
            
            m = re.search(r'[a-zA-Z]+', line)
            e = m.group(0)
            events.append((t, e))

total_mem = max(total_mem)

fig, ax = plt.subplots()
infer_mem, train_mem = map(np.array, [infer_mem, train_mem])
invert_train_mem = total_mem - train_mem 


ax.plot(timeline, infer_mem, label='Infer', linewidth=1)
ax.fill_between(timeline, 0, infer_mem, alpha=0.1)

ax.plot(timeline, invert_train_mem, label='Train', linewidth=1)
ax.fill_between(timeline, invert_train_mem, total_mem, alpha=0.1)

if args.event:
    ax.vlines([t for t, e in events], 0.0, total_mem, color='black', linewidth=0.5, linestyles='dashed')


# print([t for t, e in events])
# ax.vlines([t for t, e in events], 0, total_mem, 
#           color='black', linewidth=0.5, linestyles='dashed')


ax.set_ylim(0, total_mem)
ax.set_xlim(0,
            min(max(timeline), end_time - begin_time))

ax.set_xlabel("Time (ms)", fontsize=14)
ax.set_ylabel("GPU Memory (MB)",fontsize=14)

plt.savefig('memory-trace.svg')
