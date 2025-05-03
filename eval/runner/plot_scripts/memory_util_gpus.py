from typing import cast
import matplotlib.pyplot as plt
import pandas as pd
import io
import re
import plt_comm as pc

pc.set_color_cycle()

sec_duration = 300
duration = 300 * 1000  # ms

def extract_data(file_path):
    max_device_num = 32
    memory_info_strs = [None] * max_device_num
    parse_mem = False
    device_id = None
    start_time = None

    with open(file_path) as f:
        for line in f.readlines():
            start_time_match = re.search(r'start time stamp (\d+\.\d+) delay before profile (\d+\.\d+) sec', line)
            if start_time_match:
                start_time = float(start_time_match.group(1)) + float(start_time_match.group(2)) * 1000

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
    dfs = []
    memory_transform = lambda x: float(re.search(r'(\d+\.\d+) Mb', x).group(1)) / 1024
    for i in range(num_devices):
        df = pd.read_csv(io.StringIO(memory_info_strs[i]), sep=',', header=0)
        df.columns = df.columns.str.strip()
        df = df[df['TimeStamp'] <= duration + start_time]
        df = df[df['TimeStamp'] >= start_time]
        df['TimeStamp'] = (df['TimeStamp'] - start_time) / 1000  # ms to s
        for col in ['InferMem', 'TrainAllMem', 'ColdCache', 'ColdCacheSize']:
            df[col] = df[col].apply(memory_transform)
        dfs.append(df)
    return dfs

def plot_memory_usage(dataframes, axes, set_xticks=False):
    lines = []
    for i, (ax, df) in enumerate(zip(axes, dataframes)):
        infer_minus_cold = df['InferMem'] - df['ColdCache']
        timestamps = df['TimeStamp']

        l1 = ax.fill_between(timestamps, 0, infer_minus_cold, 
                             label='Inference', alpha=0.5, color='C0', lw=0.5*pc.lw)
        l2 = ax.fill_between(timestamps, infer_minus_cold, 
                             infer_minus_cold + df['ColdCacheSize'], 
                             label='Cache', alpha=0.3, color='gray', lw=0.5*pc.lw)
        l3 = ax.fill_between(timestamps, 
                             infer_minus_cold + df['ColdCacheSize'],
                             infer_minus_cold + df['ColdCacheSize'] + df['TrainAllMem'], 
                             label="Train", alpha=0.5, color='C1', lw=0.5*pc.lw)

        ax.tick_params(axis='both', which='both', labelsize=pc.tick_ftsz, 
                       width=pc.lw, size=2, pad=pc.tick_pad)
        ax.set_xlim(0, sec_duration)
        ax.set_ylim(0, 16)
        
        if i == 0:
            xrange = range(0, sec_duration-1, 50)
            tic = [str(i) for i in xrange] if set_xticks else [''] * len(xrange)
            ax.set_xticks(xrange, tic)
        else:
            xrange = range(0, sec_duration+1, 50)
            tic = [str(i) for i in xrange] if set_xticks else [''] * len(xrange)
            ax.set_xticks(xrange, tic)

        if i == 0:
            lines.extend([l1, l2, l3])

    return lines

bal_dfs = extract_data('memory_util_gpus_bal.dat')
imbal_dfs = extract_data('memory_util_gpus_imbal.dat')

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(85 * pc.mm,40 * pc.mm),
                         sharey=True)
fig.subplots_adjust(left=0.10, right=0.98, bottom=0.2, 
                    top=0.865, hspace=0.0, wspace=0.03) 

lines_balance = plot_memory_usage(bal_dfs, axes[0, :], set_xticks=False)
lines_imbalance = plot_memory_usage(imbal_dfs, axes[1, :], set_xticks=True)

# Add a single legend for the entire figure
fig.legend(lines_balance, ['Inference', 'Reservation', 'Training'], 
           loc='upper center', bbox_to_anchor=(0.55, 1.01), 
           columnspacing=2, 
           # handlelength=0.8, 
        #    handletextpad=0.4,
           ncol=3, fontsize=pc.label_ftsz, facecolor='none'
        ).get_frame().set_linewidth(pc.lw)

for ax in axes[1, :]:
    ax.set_xlabel('Timeline (sec)', fontsize=pc.label_ftsz, labelpad=pc.label_pad)

for ax, title in zip(axes[:, 0], ['w/ BD', 'w/o BD']):
    ax.set_ylabel(f'{title}\nMem. (GB)', fontsize=pc.label_ftsz, labelpad=pc.label_pad)

for ax in axes.flatten():
    ax.set_yticks([0, 10])

for ax, title in zip(axes[1, :], ['GPU 0', 'GPU 1']):
    # ax.set_title(title, fontsize=pc.label_ftsz, pad=pc.label_pad, loc='left', x=0.05)
    ax.text(0.5, -0.41, title, fontsize=pc.label_ftsz+1, transform=ax.transAxes, va='top', ha='center')



for ax in axes.flatten():
    plt.setp(ax.spines.values(), linewidth=pc.lw)
    

# add thpt

data = pd.read_csv('memory_util_gpus_thpt.dat', sep='\s+', index_col=0, comment='#')


for i, is_bal in enumerate(['BAL', 'IMBAL']):
    for gpu_id in [0, 1]:
        gpu_value = data.loc[is_bal, f'GPU{gpu_id}']
        ax = axes[i, gpu_id]
        ax.text(0.15, 0.84, f'Thpt={gpu_value:.1f} sam/s', fontsize=pc.label_ftsz, transform=ax.transAxes)
plt.savefig("memory_util_gpus.pdf")
