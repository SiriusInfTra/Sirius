import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plt_comm as pc

pc.set_color_cycle()

df = pd.read_csv("multi_gpu.dat", sep='\s+', index_col=[0], header=0, comment='#')

# workload = df.head(1).columns.to_list()
systems = ['TaskSwitch', 'SP-50', 'SP-75', 'UM+MPS', 'Sirius', 'Infer-Only']

fig, ax = plt.subplots(1, 3, figsize=(85*pc.mm, 30*pc.mm))
fig.subplots_adjust(left=0.10, right=0.98, top=0.78, bottom=0.05, wspace=0.5)

x = np.arange(len(systems))
handles = []

df['SLO'] = df['SLO'] * 100

for i in range(3):
    width = 0.35
    margin = 0 # 0.05
    offset = 0

    # type = 'SLO' if i == 0 else 'Train'
    type = ['Infer', 'SLO', 'Train'][i]
    for j in range(len(systems)):
        s = systems[j]
        rects = ax[i].bar([j+offset+margin], [df.loc[s, type]], width=width, label=s, color=f'C{j}')
        if i == 0:
            handles.append(rects)
        if i == 2 and s == 'Infer-Only':
            ax[i].scatter([j+offset+margin], [df.loc[s, type]+20], marker='x', color=f'C{j}', s=10)

    if i == 0:
        ax[i].set_ylabel('Infer P99 (ms)', fontsize=pc.label_ftsz, labelpad=pc.label_pad)
        ax[i].set_yscale('log')
    elif i == 1:
        ax[i].set_ylabel('Infer SLO (%)', fontsize=pc.label_ftsz, labelpad=pc.label_pad)
        ax[i].set_yticks([0, 50, 100])
        # ax[i].set_yscale('log')
    else:
        ax[i].set_ylabel('Train thpt (sam/s)', fontsize=pc.label_ftsz)
        ylim = ax[i].get_ylim()
        ax[i].set_ylim([-15, ylim[1]])

    plt.setp(ax[i].spines.values(), linewidth=pc.lw)
    ax[i].tick_params(axis='both', which='both', labelsize=pc.tick_ftsz+1, 
                    pad=pc.tick_pad, size=2, width=pc.lw,)
    # ax[i].set_xticks(np.arange(len(systems)), systems)
    ax[i].set_xticks([0, 1, 2, 3, 4, 5], ['', '', '', '', '', ''])

fig.legend(handles, systems, fontsize=pc.label_ftsz, loc='upper right', ncol=6,
           columnspacing=0.6, handlelength=0.8,
           bbox_to_anchor=(0.995, 1)).get_frame().set_linewidth(pc.lw)

plt.savefig('multi_gpu.pdf')