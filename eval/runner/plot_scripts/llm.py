import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plt_comm as pc

pc.set_color_cycle()

df = pd.read_csv("llm.dat", sep='\s+', index_col=[0], header=0, comment='#')

systems = ['SP-50', 'SP-75', 'Sirius', 'Infer-Only']


fig, ax = plt.subplots(1, 3, figsize=(85*pc.mm, 30*pc.mm))
fig.subplots_adjust(left=0.10, right=0.98, top=0.78, bottom=0.05, wspace=0.5)

columns = ['TTFT-SLO', 'TBT-SLO', 'Train']
x = np.arange(len(systems))
handles = []

df['TTFT-SLO'] = df['TTFT-SLO'] * 100
df['TBT-SLO'] = df['TBT-SLO'] * 100

for i in range(3):
    width = 0.3
    margin = 0
    offset = 0
    for j, s in enumerate(systems):
        rects = ax[i].bar([j+offset+margin], [df.loc[s, columns[i]]],
                          width=width, label=s, color=pc.sys_color(s))
        if i == 0:
            handles.append(rects)
        if i == 2 and s == 'Infer-Only':
            ax[i].scatter([j+offset+margin], [df.loc[s, columns[i]]+2], marker='x', 
                          color=pc.sys_color(s), s=10)

    if i == 0:
        ax[i].set_ylabel('TTFT SLO (%)', fontsize=pc.label_ftsz, labelpad=-1)
        ax[i].set_yticks([0, 50, 100])
    elif i == 1:
        ax[i].set_ylabel('TBT SLO (%)', fontsize=pc.label_ftsz, labelpad=-1)
        ax[i].set_yticks([0, 50, 100])
    else:
        ax[i].set_ylabel('Train thpt (sam/s)', fontsize=pc.label_ftsz, labelpad=pc.label_pad)
        ylim = ax[i].get_ylim()
        ax[i].set_ylim([-3, ylim[1]])

    ax[i].set_xticks([0, 1, 2, 3], ['', '', '', ''])
    ax[i].set_xlim([-0.5, 3.5])
    plt.setp(ax[i].spines.values(), linewidth=pc.lw)
    ax[i].tick_params(axis='both', labelsize=pc.tick_ftsz+1,
                      pad=pc.tick_pad, size=2, width=pc.lw)

fig.legend(handles, systems, fontsize=pc.label_ftsz, loc='upper center', ncol=4,
        #    columnspacing=0.6, 
           handlelength=1.5,
           bbox_to_anchor=(0.5, 1)).get_frame().set_linewidth(pc.lw)

fig.savefig('llm.pdf')