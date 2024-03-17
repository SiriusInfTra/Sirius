import matplotlib.pyplot as plt
import numpy as np
import argparse
import re

app = argparse.ArgumentParser("infering memory")
app.add_argument('-l', type=str)
app.add_argument('-o', type=str)

args = app.parse_args()
log = args.l
outdir = args.o

timeline = []
infering_memory = []


max_infer_memory = 0
parse_infering_memory = False
with open(log) as f:
    for line in f.readlines():
        m = re.search(r'Infer (\d+\.\d+) Mb', line)
        if m is not None:
            max_infer_memory = max(max_infer_memory, float(m.group(1)))

        if '[Infering Model Memory Info]' in line:
            parse_infering_memory = True
            continue

        if len(line.strip()) == 0:
            parse_infering_memory = False

        if parse_infering_memory:
            m = re.search(r'(\d+(\.\d+)?): (\d+): (\d+)', line)
            timeline.append(float(m.group(1)))
            infering_memory.append(int(m.group(4))/1024/1024)

timeline = np.array(timeline)
timeline -= np.min(timeline)

# print(infering_memory)
print(max_infer_memory)
print(max(infering_memory))

fig, ax = plt.subplots()

ax.fill_between(timeline, 0, max_infer_memory, alpha=0.1)
ax.plot(timeline, infering_memory, lw=0.2, label='Infering Memory')
ax.fill_between(timeline, 0, infering_memory, alpha=0.3)

ax.set_xlim(0, max(timeline))
ax.set(xlabel='Time (ms)', ylabel='Memory (Mb)',)
plt.legend()
plt.savefig(f'{outdir}/infering_memory.svg')

