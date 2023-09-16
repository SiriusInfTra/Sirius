import re, sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

app = argparse.ArgumentParser("memory trace")
app.add_argument('-l', type=str)
app.add_argument('-b', type=int, default=0)
app.add_argument('-e', type=int, default=3600*1e3)
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
            timeline.append(float(m.group(1)))

            m = re.search(r'Infer (\d+\.\d+) Mb', line)
            infer_mem.append(float(m.group(1)))

            m = re.search(r'Train (\d+\.\d+) Mb', line)
            train_mem.append(float(m.group(1)))

            m = re.search(r'Total (\d+\.\d+) Mb', line)
            total_mem.append(float(m.group(1)))

        if paser_ev:
            m = re.search(r'(\d+(\.\d+)?):', line)
            t = float(m.group(1))
            
            m = re.search(r'[a-zA-Z]+', line)
            e = m.group(0)
            events.append((t, e))

total_mem = max(total_mem)

fig, ax = plt.subplots()
infer_mem, train_mem = map(np.array, [infer_mem, train_mem])
invert_train_mem = total_mem - train_mem 


ax.plot(timeline, infer_mem, label='Infer')
ax.plot(timeline, invert_train_mem, label='Train')
ax.vlines([t for t, e in events], 0.0, total_mem, color='black', linewidth=0.5, linestyles='dashed')


# print([t for t, e in events])
# ax.vlines([t for t, e in events], 0, total_mem, 
#           color='black', linewidth=0.5, linestyles='dashed')


ax.set_ylim(0, total_mem)
ax.set_xlim(max(min(timeline), begin_time), 
            min(max(timeline), end_time))

plt.savefig('memory-trace.svg')
