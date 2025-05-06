import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import plt_comm as pc
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--single-gpu', action='store_true', default=False)
parser.add_argument('--multi-gpu', action='store_true', default=False)
args = parser.parse_args()

if not args.single_gpu and not args.multi_gpu:
    print("Please specify --single-gpu or --multi-gpu")
    exit(1)

# ============================================================

pc.set_color_cycle()

def get_breakdown_info(filename):
    train_adjust = []
    infer_alloc = []
    infer_load_param = []
    infer_pipeline_exec = []
    infer_avg_exec = None

    parse_train_adjust = False
    parse_infer_alloc = False
    parse_infer_load_param = False
    parse_infer_pipeline_exec = False
    parse_infer_avg_exec = False

    def parse_off():
        nonlocal parse_train_adjust, parse_infer_alloc, parse_infer_load_param, \
            parse_infer_pipeline_exec, parse_infer_avg_exec
        parse_train_adjust = False
        parse_infer_alloc = False
        parse_infer_load_param = False
        parse_infer_pipeline_exec = False
        parse_infer_avg_exec = False

    with open(filename, 'r') as F:
        for line in F.readlines():
            if line.strip() == "" or line.strip().startswith('#'):
                parse_off()
                continue
            if 'TrainAdjust' in line:
                parse_off()
                parse_train_adjust = True
                continue
            elif 'InferAllocStorage' in line:
                parse_off()
                parse_infer_alloc = True
                continue
            elif 'InferLoadParam' in line:
                parse_off()
                parse_infer_load_param = True
                continue
            elif 'InferPipelineExec' in line:
                parse_off()
                parse_infer_pipeline_exec = True
                continue
            elif 'InferExec' in line:
                parse_off()
                parse_infer_avg_exec = True
                continue
            
            t = line.strip().split()
            t = list(map(float, t))
            if parse_train_adjust:
                train_adjust.extend(t)
            elif parse_infer_alloc:
                infer_alloc.extend(t)
            elif parse_infer_load_param:
                infer_load_param.extend(t)
            elif parse_infer_pipeline_exec:
                infer_pipeline_exec.extend(t)
            elif parse_infer_avg_exec:
                infer_avg_exec = t[0]

    train_adjust = sorted(train_adjust)[:int(len(train_adjust) * 0.99 + 0.5)]
    infer_alloc = sorted(infer_alloc)[:int(len(infer_alloc) * 0.99 + 0.5)]

    return {'train_adjust': train_adjust,
            'infer_alloc': infer_alloc,
            'infer_load_param': infer_load_param,
            'infer_pipeline_exec': infer_pipeline_exec,
            'infer_avg_exec': infer_avg_exec}

if args.single_gpu:
    sirius_single_gpu = get_breakdown_info("adjust_breakdown_sirius_single_gpu.dat")
    naive_single_gpu = get_breakdown_info("adjust_breakdown_naive_single_gpu.dat")
    datas = [
        {'sirius': sirius_single_gpu, 'naive': naive_single_gpu},
    ]
else:
    sirius_multi_gpu = get_breakdown_info("adjust_breakdown_sirius_multi_gpu.dat")
    naive_multi_gpu = get_breakdown_info("adjust_breakdown_naive_multi_gpu.dat")
    datas = [
        {'sirius': sirius_multi_gpu, 'naive': naive_multi_gpu}
    ]    


mm = 1/25.4
fig, ax = plt.subplots(1, 3, figsize=(85*mm,32*mm), gridspec_kw={'width_ratios': [2, 2, 1.5]})
fig.subplots_adjust(left=0.15, right=0.97, bottom=0.20, top=0.82, wspace=0.55, hspace=0.25)
label_ftsz = 6
tick_ftsz = 5

# to check

handle = []
for i in range(len(datas)):
    data = datas[i]
    row_name = 'Single GPU' if args.single_gpu else 'Multi GPU'
    sirius = data['sirius']
    naive = data['naive']

    print(f'{row_name} train_adjust p99, {np.max(sirius["train_adjust"])} | {np.max(naive["train_adjust"])}')

    l1 = ax[0].ecdf(naive['train_adjust'], label='Naive', lw=1)
    l2 = ax[0].ecdf(sirius['train_adjust'], label='Sirius', lw=1)

    # print(sirius['train_adjust'])

    ax[0].set_xscale('log')
    ax[0].set_ylim([0, 0.99])
    ytics = [0,  0.5,  0.99]
    ax[0].set_yticks(ytics)
    ax[0].set_xlabel("Latency (ms)", fontsize=label_ftsz, labelpad=1)
    ax[0].set_ylabel(f"Train Adjust CDF", fontsize=label_ftsz, labelpad=1)
    ax[0].annotate(row_name, xy=(0, 0.99/2), xycoords=ax[0].yaxis.label, 
                      xytext=(-ax[0].yaxis.labelpad - 6, 0), textcoords='offset points',
                      fontsize=label_ftsz+1, ha='center', va='center',rotation=90)

    print(f'{row_name} infer_alloc p99, {np.max(sirius["infer_alloc"])} | {np.max(naive["infer_alloc"])}')

    ax[1].ecdf(naive['infer_alloc'], label='Naive', lw=1)
    ax[1].ecdf(sirius['infer_alloc'], label='Sirius', lw=1)
    ax[1].set_xscale('log')
    ax[1].set_ylim([0, 0.99])
    ax[1].set_yticks(ytics)
    ax[1].set_xlabel("Latency (ms)", fontsize=label_ftsz, labelpad=1)
    ax[1].set_ylabel("Infer Alloc CDF", fontsize=label_ftsz, labelpad=1)

    sirius_avg_pipeline_exec = np.mean(sirius['infer_pipeline_exec'])
    naive_avg_loading = np.mean(naive['infer_load_param'])
    sirius_avg_loading = np.mean(sirius['infer_load_param'])
    print(sirius_avg_pipeline_exec)
    l3 = ax[2].bar([0], [naive_avg_loading], label='Naive', width=0.8)
    l4 = ax[2].bar([1], [sirius_avg_pipeline_exec-sirius['infer_avg_exec']], label='Sirius')
    print(f"{row_name} load overhead, {naive_avg_loading:.2f} | {sirius_avg_pipeline_exec-sirius['infer_avg_exec']:.2f}")
    print(f"{row_name} load time {naive_avg_loading:.2f} | {sirius_avg_loading:.2f}")
    ax[2].set_xlim([-0.8, 1.8])
    ax[2].set_xticks([])
    ax[2].set_ylabel("Infer Loading\nOverhead (ms)", fontsize=label_ftsz, labelpad=1)

    ax[0].tick_params(axis='both', which='both', width=0.5, labelsize=tick_ftsz, size=2, pad=1)
    ax[1].tick_params(axis='both', which='both', width=0.5, labelsize=tick_ftsz, size=2, pad=1)
    ax[2].tick_params(axis='both', which='both', width=0.5, labelsize=tick_ftsz, size=2, pad=1)
    plt.setp(ax[0].spines.values(), linewidth=0.5)
    plt.setp(ax[1].spines.values(), linewidth=0.5)
    plt.setp(ax[2].spines.values(), linewidth=0.5)

    if handle == []:
        handle = [l1, l2, l3, l4]

fig.legend(handle, ['Naive', 'Sirius', 'Naive', 'Sirius'], bbox_to_anchor=(0.5, 0.92),
           ncol=4, loc='center', fontsize=label_ftsz, ).get_frame().set_linewidth(0.5)


if args.single_gpu:
    plt.savefig("adjust_breakdown_single_gpu.pdf")
if args.multi_gpu:
    plt.savefig("adjust_breakdown_multi_gpu.pdf")
