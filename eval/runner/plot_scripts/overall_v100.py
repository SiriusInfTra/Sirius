import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.path as mpath
from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerPathCollection, HandlerLine2D

import plt_comm as pc

pc.set_color_cycle()

df = pd.read_csv("overall_v100.dat", sep='\s+', index_col=[0, 1], header=0, comment='#')

# workload = ['Uniform-A', 'Uniform-B', 'Uniform-C', 'Skew-C', 'MAF']
workload = ['Light', 'Heavy', 'Burst', 'Skewed', 'MAF']
systems = ['TaskSwitch', 'SP-F', 'SP-I', 'UM+MPS', 'Sirius', 'Infer-Only']


# systems = list(set([x[0] for x in df.index.to_list()]))

# fig, (ax1, ax2, ax4) = plt.subplots(3, 1, figsize=(85*pc.mm, 60*pc.mm), 
#                                     gridspec_kw={'height_ratios': [0.5, 0.5, 1], 'hspace': [0.1, 0.1, 0.1]})

adjust = {
    'left': 0.09, 'right': 0.99,
}

bottom = 0.04
top = 0.92
mid1 = 2*(top-bottom)/3+bottom
mid2 = (top-bottom)/3+bottom
margin = 0.015
gs_1 = plt.GridSpec(2, 1, hspace=0.15,  top=top-margin, bottom=mid1+margin, **adjust, 
                    height_ratios=[0.3, 0.7])
gs_2 = plt.GridSpec(1, 1, hspace=0.3,  top=mid1-margin, bottom=mid2+margin, **adjust)
gs_3 = plt.GridSpec(1, 1, top=mid2-margin, bottom=bottom+margin, **adjust)
fig = plt.figure(figsize=(85*pc.mm, 70*pc.mm))

ax1 = fig.add_subplot(gs_1[0, 0])
ax2 = fig.add_subplot(gs_1[1, 0])
ax3 = fig.add_subplot(gs_2[0, 0])
ax4 = fig.add_subplot(gs_3[0, 0])

# fig.subplots_adjust(left=0.1, right=0.99, top=0.85, bottom=0.08,)

# ax2.sharex(ax1)
ax = [[ax2, ax1], [ax3], [ax4]]

x = np.arange(len(workload))

handles = []
star_handle = None

for s in systems:
    df.loc[(s, 'SLO'), :] = df.loc[(s, 'SLO'), :] * 100


types = ['Infer', 'SLO', 'Train']
for i in range(3): # infer, train
    # type = 'Infer' if i == 0 else 'Train'
    type = types[i]

    width = 0.05
    margin = 0.075
    offset = 0
    for j in range(len(systems)):
        s = systems[j]
        rects = ax[i][0].bar(x + offset + margin, df.loc[(s, type), :], width)
        if i == 0:
            # if s != 'Sirius':
            handles.append(rects)
            ax[i][1].bar(x + offset + margin, df.loc[(s, type), :], width)
        if type == 'Train' and 'Infer-Only' == s:
            ax[i][0].scatter(x+offset+margin, 5+np.zeros(len(workload)), marker='x', color=f'C{j}', s=10)
            
        # if type == 'Infer' and 'Sirius' == s:
        #     star_handle = ax[i][0].scatter(x+offset+margin, np.zeros(len(workload))-14, marker='*', color=f'C{j}', s=16, 
        #                                    clip_on=False, lw=0.5)
        #     handles.append(star_handle)

        offset += width + margin

    if i == 0:
        ax[i][0].set_ylabel("       Infer P99 (ms)", fontsize=pc.label_ftsz, labelpad=pc.label_pad)
        broken_axis_y = 160
        ax[i][0].set_ylim(0, broken_axis_y)
        ax[i][0].spines['top'].set_visible(False)
        ylim = ax[i][1].get_ylim()
        ylim = (broken_axis_y, ylim[1])
        ax[i][1].set_yscale("log")
        ax[i][1].set_ylim(ylim)
        ax[i][1].spines['bottom'].set_visible(False)

        d = .5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=3,
                    linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax[i][0].plot([0, 1], [1, 1], transform=ax[i][0].transAxes, **kwargs)
        ax[i][1].plot([0, 1], [0, 0], transform=ax[i][1].transAxes, **kwargs)
    elif i == 1:
        ax[i][0].set_ylabel("Infer SLO (%)", fontsize=pc.label_ftsz, labelpad=pc.label_pad)
        ax[i][0].set_ylim(0, 100)
        ax[i][0].set_yticks([0, 50, 100])
        # ax[i][0].set_xticks(range(len(workload)), [''] * len(workload))
        # ax[i][0].set_xticks([])
    else:
        ax[i][0].set_ylabel("Train thpt (sam/s)", fontsize=pc.label_ftsz, labelpad=pc.label_pad)
        ylim = ax[i][0].get_ylim()
        ylim = (-5, ylim[1])
        ax[i][0].set_ylim(ylim)

    if i == 2:
        ax[i][0].set_xticks(x+offset/2-width/2+margin/2, workload)
    else:
        ax[i][0].set_xticks(x+offset/2-width/2+margin/2, [''] * len(workload))
    ax[i][0].tick_params(axis='both', which='both', labelsize=pc.tick_ftsz+1, 
                      pad=pc.tick_pad, size=2, width=pc.lw)
    if i == 2:
        ax[i][0].tick_params(axis='x', which='both', labelsize=pc.tick_ftsz+1.5)
    if i == 0:
        ax[i][1].set_xticks([])
        ax[i][0].tick_params(axis='both', which='both', labelsize=pc.tick_ftsz+1, 
                      pad=pc.tick_pad, size=2, width=pc.lw)
        ax[i][1].tick_params(axis='both', which='both', labelsize=pc.tick_ftsz+1, 
                      pad=pc.tick_pad, size=2, width=pc.lw)
    
    
    xs = x+offset/2-width/2+margin/2
    for a, b in zip(xs[1:], xs[:-1]):
        c = (a+b)/2
        ylim = ax[i][0].get_ylim()
        ax[i][0].vlines(c, ylim[0], ylim[1], color='black', lw=0.8*pc.lw, linestyles='--')
        if i == 0:
            ylim = ax[i][1].get_ylim()
            ax[i][1].vlines(c, ylim[0], ylim[1], color='black', lw=0.8*pc.lw, linestyles='--')

# print(systems)
# order = [0, 4, 1, 5, 2, 3]
# order = [0, 1, 2, 3, 4, 5]

# new_handles = [handles[i] for i in order]
# new_systems = [systems[i] for i in order]

# we use SP-number instead of SP-{I,F}
new_systems = []
for s in systems:
    if s == 'SP-I':
        label = 'SP-75'
    elif s == 'SP-F':
        label = 'SP-50'
    else:
        label = s
    new_systems.append(label)


def update_lgnd(handle, orig):
    handle.update_from(orig)
    handle.set_sizes([24])
    handle.set_offsets([(3.5, 1.8)])

fig.legend(handles, new_systems, fontsize=pc.label_ftsz, bbox_to_anchor=(0.53, 0.91),
           loc='lower center', ncol=6, columnspacing=0.6, handlelength=0.8,
        #    handler_map={PathCollection : HandlerPathCollection(update_func=update_lgnd)}
        ).get_frame().set_linewidth(pc.lw)

plt.savefig("overall_v100.pdf")

slo_delta = []
for w in workload:
    for s in systems:
        if s == 'Infer-Only' or s == 'Sirius':
            continue
        v = df.loc[('Sirius', 'SLO'), w] - df.loc[(s, 'SLO'), w]
        slo_delta.append(v)

print('SLO', len(slo_delta), np.mean(slo_delta), np.max(slo_delta))

train_delta = []
for w in workload:
    for s in systems:
        if s == 'Infer-Only' or s == 'Sirius':
            continue
        if df.loc[(s, 'Train'), w] == 0:
            continue
        v = df.loc[('Sirius', 'Train'), w] / df.loc[(s, 'Train'), w]
        train_delta.append(v)

print('Train', len(train_delta), np.mean(train_delta), np.max(train_delta))

    