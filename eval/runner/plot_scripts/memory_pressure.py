import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

import plt_comm as pc

pc.set_color_cycle()

ltc = pd.read_csv("memory_pressure_ltc.dat", 
                 sep=',', header=0, index_col=False, comment='#')

thpt = pd.read_csv("memory_pressure_thpt.dat",
                   sep=',', header=0, index_col=False, comment='#')

fig, ax = plt.subplots(figsize=(85*pc.mm, 30*pc.mm))
fig.subplots_adjust(left=0.09, right=0.91, top=0.9, bottom=0.21)

# print(df)

start = ltc['start_timestamp'].min()

thpt = thpt[thpt['start_timestamp'] >= start]
thpt['ms'] = thpt['start_timestamp'] - start
thpt = thpt[thpt['ms'] < 24*60*1000]
thpt['sec'] = thpt['ms'].transform(lambda x: int(x/5000))

ltc['ms'] = ltc['start_timestamp'] - start
ltc['sec'] = ltc['ms']//5000

# df = df.sort_values(by='ms')

# df['ms'] = df['end_timestamp'] - df['end_timestamp'].min()

# df['avg_ltc'] = df.groupby('sec')['ltc'].transform('max')
# def get_p99(df):
#     return df.quantile(0.99)

# print(df.groupby('sec')['ltc'].mean().index)
ltc = ltc.groupby('sec')['ltc'].mean()
thpt = thpt.groupby('sec')['batch_size'].sum() / 5
# print(thpt)
# ltc_err = df.groupby('min')['ltc'].std()
# ltc = df.groupby('sec')['ltc'].mean()
# ltc_err = df.groupby('sec')['ltc'].std()
# ltc = df.groupby('min')['ltc'].quantile(0.99)

# print(thpt.index)

all_ltc = np.zeros(24*60//5)
all_thpt = np.zeros(24*60//5)
# print(df['sec'].groupby('sec')['sec'])
all_ltc[ltc.index] = ltc
all_thpt[thpt.index] = thpt

handles = []

l1 = ax.plot(np.arange(len(all_ltc)), all_ltc, lw=pc.lw, color='C0')
ax.set_xlim(0, 24 * 60 // 5)
ax.set_xticks(np.arange(0, 24 * 60 // 5 + 1, 6 * 60 // 5), [f'{i}' for i in range(0, 24+6, 6)])
ax.set_xlabel("Timeline (minute)", fontsize=pc.label_ftsz, labelpad=pc.label_pad)
ax.set_ylabel("Latency (ms)", fontsize=pc.label_ftsz, labelpad=pc.label_pad)
# ax.fill_between(np.arange(len(ltc)), ltc-ltc_err, ltc+ltc_err, alpha=0.5)
ax_ylim = ax.get_ylim()
# print(ax_ylim)
# ax.set_ylim(10, ax_ylim[1])
ax.set_ylim(10, 500)
ax.set_yscale('log')

ax2 = ax.twinx()
ax2.set_ylabel("Throughput (sam/s)", fontsize=pc.label_ftsz, labelpad=pc.label_pad)
l2 = ax2.plot(np.arange(len(all_thpt)), all_thpt, lw=pc.lw, color='C1')
ax2_ylim = ax2.get_ylim()
ax2.set_ylim(0, ax2_ylim[1])


# ax.annotate(f'')

ax.tick_params(axis='both', which='both', labelsize=pc.tick_ftsz, pad=pc.tick_pad, size=2, width=pc.lw)
ax2.tick_params(axis='both', which='both', labelsize=pc.tick_ftsz, pad=pc.tick_pad, size=2, width=pc.lw)
plt.setp(ax.spines.values(), linewidth=pc.lw)
plt.setp(ax2.spines.values(), linewidth=pc.lw)

# plt.vlines(420//5, ax2_ylim[0], ax_ylim[1], color='black', lw=pc.lw, linestyle='--')
ax.annotate("Reaching\nadjustment limit", xy=(420//5, 100), xytext=(3, 3), 
            textcoords='offset pixels', fontsize=pc.label_ftsz, color='black',
            ha='left')
            # va='center',
            # arrowprops=dict(facecolor='black', arrowstyle='simple', lw=pc.lw, color='black'))


plt.vlines(420//5, ax2_ylim[0], ax_ylim[1], color='black', lw=1.5*pc.lw, linestyle='--')
# ax.set_yticks([10, 100])
# print(df['ltc'])

fig.legend([l1[0], l2[0]], ['Infer', 'Train'], 
           loc='lower right', bbox_to_anchor=(0.9, 0.2),
           fontsize=pc.label_ftsz).get_frame().set_linewidth(pc.lw)

fig.savefig("memory_pressure.pdf")