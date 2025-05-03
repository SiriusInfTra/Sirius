import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plt_comm as pc

pc.set_color_cycle()

infer_idle_ms_df = pd.read_csv("ablation_infer_idle_time.dat", 
                sep='\s+', header=0, index_col=False, comment='#',
                dtype={'MaxIdleMs': str})

cache_size_df = pd.read_csv("ablation_cache_size.dat", 
                 sep='\s+', header=0, index_col=False, comment='#')


fig, ax = plt.subplots(2, 2, figsize=(85*pc.mm, 60*pc.mm))
fig.subplots_adjust(left=0.095, right=0.9, top=0.82, bottom=0.1, wspace=0.52, hspace=0.44)

x = np.arange(len(infer_idle_ms_df['MaxIdleMs']))
x_ticks = [str(x).upper() for x in infer_idle_ms_df['MaxIdleMs'].to_list()]

handles_infer = []
handles_train = []

for i in range(2):
    ax1 = ax[0][i]
    ax2 = ax1.twinx()

    if i == 0:
        l1 = ax1.plot(x, infer_idle_ms_df['P99'], label='P99 Latency', color='C0', marker='o', lw=2*pc.lw, markersize=3)
        ax1.set_xlabel("Idle Model Liveness Time (sec)", fontsize=pc.label_ftsz, labelpad=pc.label_pad)
        ax1.set_ylabel("Infer P99 (ms)",  fontsize=pc.label_ftsz, labelpad=pc.label_pad)
        ylim = ax1.get_ylim()
        ax1.set_ylim([0, ylim[1]])
        ax1.set_xticks(x, x_ticks)

        l2 = ax2.plot(x, infer_idle_ms_df['ColdStart'], label='ColdStart%', color='C1', marker='v', lw=2*pc.lw, markersize=3)
        ax2.set_ylabel("Cold-Starts Pct. (%)", fontsize=pc.label_ftsz, labelpad=pc.label_pad)
        ylim = ax2.get_ylim()
        ax2.set_ylim([0, ylim[1]])

        handles_infer.append(l1[0])
        handles_infer.append(l2[0])

    elif i == 1:
        l1 = ax1.plot(x, infer_idle_ms_df['Thpt'], label='Throughput', color='C3', marker='o', lw=2*pc.lw, markersize=3)
        ax1.set_xlabel("Idle Model Liveness Time (sec)", fontsize=pc.label_ftsz, labelpad=pc.label_pad)
        ax1.set_ylabel("Train Thpt (sam/s)",  fontsize=pc.label_ftsz, labelpad=pc.label_pad)
        ylim = ax1.get_ylim()
        ax1.set_ylim([0, ylim[1]])
        ax1.set_xticks(x, x_ticks)


        l2 = ax2.plot(x, infer_idle_ms_df['WasteCompute'], label='Wasted Computing', color='C4', marker='v', lw=2*pc.lw, markersize=3)
        ax2.set_ylabel("Wasted Computing\nPct. (%)", fontsize=pc.label_ftsz, labelpad=pc.label_pad)
        ylim = ax2.get_ylim()
        ax2.set_ylim([0, ylim[1]])

        handles_train.append(l1[0])
        handles_train.append(l2[0])

    plt.setp(ax1.spines.values(), linewidth=pc.lw)
    plt.setp(ax2.spines.values(), linewidth=pc.lw)
    ax1.tick_params(axis='both', which='both', labelsize=pc.tick_ftsz, pad=pc.tick_pad, size=2, width=pc.lw)
    ax2.tick_params(axis='both', which='both', labelsize=pc.tick_ftsz, pad=pc.tick_pad, size=2, width=pc.lw)

x_ticks = cache_size_df['CacheSize'] / 2
x = np.arange(len(x_ticks))

for i in range(2):
    ax1 = ax[1][i]
    ax2 = ax1.twinx()

    if i == 0:
        l1 = ax1.plot(x, cache_size_df['P99'], label='P99 Latency', color='C0', marker='o', lw=2*pc.lw, markersize=3)
        ax1.set_xlabel("Reallocation WM (GB)", fontsize=pc.label_ftsz, labelpad=pc.label_pad)
        ax1.set_ylabel("Infer P99 (ms)",  fontsize=pc.label_ftsz, labelpad=pc.label_pad)
        ylim = ax1.get_ylim()
        ax1.set_ylim([0, ylim[1]])
        # ax1.set_ylim([0, 100])
        ax1.set_xticks(x, x_ticks)

        l2 = ax2.plot(x, cache_size_df['Load'], label='Loading', color='C2', marker='v', lw=2*pc.lw, markersize=3)
        ax2.set_ylabel("Avg Loading Time (ms)", fontsize=pc.label_ftsz, labelpad=pc.label_pad)
        ylim = ax2.get_ylim()
        ax2.set_ylim([0, ylim[1]])

        handles_infer.append(l2[0])

    elif i == 1:
        l1 = ax1.plot(x, cache_size_df['Thpt'], label='Throughput', color='C3', marker='o', lw=2*pc.lw, markersize=3)
        ax1.set_ylabel("Train Thpt (sam/s)",  fontsize=pc.label_ftsz, labelpad=pc.label_pad)
        # ax1.set_xticks(x, x_ticks)
        ylim = ax1.get_ylim()
        ax1.set_ylim([0, ylim[1]])
        ax1.set_xticks(x, x_ticks)

        l2 = ax2.plot(x, cache_size_df['WasteCompute'], label='Wasted Computing', color='C4', marker='v', lw=2*pc.lw, markersize=3)
        ax1.set_xlabel("Reallocation WM (GB)", fontsize=pc.label_ftsz, labelpad=pc.label_pad)
        ax2.set_ylabel("Wasted Computing\nPct. (%)", fontsize=pc.label_ftsz, labelpad=pc.label_pad)
        ylim = ax2.get_ylim()
        ax2.set_ylim([0, ylim[1]])

    ax1.set_xticks(x, x_ticks)
    plt.setp(ax1.spines.values(), linewidth=pc.lw)
    plt.setp(ax2.spines.values(), linewidth=pc.lw)
    ax1.tick_params(axis='both', which='both', labelsize=pc.tick_ftsz, pad=pc.tick_pad, size=2, width=pc.lw)
    ax2.tick_params(axis='both', which='both', labelsize=pc.tick_ftsz, pad=pc.tick_pad, size=2, width=pc.lw)


legend = fig.legend(handles_infer + handles_train, ['P99 Latency', 'ColdStart%', 'Loading', 'Throughput', 'Wasted Computing'],
                    fontsize=pc.label_ftsz, ncol=3, bbox_to_anchor=(0.5, 1), loc='upper center')
legend.get_frame().set_linewidth(pc.lw)


plt.savefig("ablation.pdf")
