import argparse
import re
import pandas
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import transforms
import pathlib

def is_llm(model):
    return 'llama' in model.lower() or 'opt' in model.lower()

parser = argparse.ArgumentParser('Collect inference latency')
parser.add_argument('-l', '--log', type=str, required=True)
parser.add_argument('-o', '--output', type=str, required=False)
parser.add_argument('--slo-output', type=str, required=False)

args = parser.parse_args()

log_file = args.log
output_file = args.output
slo_output_file = args.slo_output

if slo_output_file is not None and pathlib.Path(slo_output_file).is_dir():
    slo_output_file = pathlib.Path(slo_output_file) / 'infer-slo.svg'


parse_ltc = False
model_name = None
ltcs_at_tp = []
time_to_first_token_ltc_at_tp = []
time_between_token_ltc_at_tp = []
with open(log_file, 'r') as F:
    for line in F.readlines():
        if line.strip() == '':
            parse_ltc = False
            continue
        # if 'InferWorker TRACE' in line:
        mat = re.search(r'\[InferWorker TRACE ([0-9a-zA-Z_\/]+)(-(\d+))?\]', line)
        if mat is not None:
            parse_ltc = True
            model_name = mat.group(1)
            continue
        if '=' in line or '[ ' in line or ']' in line:
            parse_ltc = False
            model_name = None
            continue

        if parse_ltc: 
            data = line.strip().split(',')
            start_tp = int(data[0].strip())
            end_tp = int(data[1].strip())
            if not is_llm(model_name):
                ltc = float(data[-1].strip())
                ltcs_at_tp.append((model_name, start_tp, end_tp, ltc))
            else:
                ltc = float(data[-3].strip())
                time_to_first_token_ltc = float(data[-2].strip())
                time_between_token_ltc = float(data[-1].strip())
                ltcs_at_tp.append((model_name, start_tp, end_tp, ltc))
                time_to_first_token_ltc_at_tp.append((
                    model_name, start_tp, end_tp, time_to_first_token_ltc))
                time_between_token_ltc_at_tp.append((
                    model_name, start_tp, end_tp, time_between_token_ltc))
            # start_tps.append(start_tp)
            # ltcs.append(ltc)

if slo_output_file is not None:
    std_ltcs = {
        'resnet152': 9.358644,
        'densenet161': 12.586594,
        'efficientnet_v2_s': 5.287647,
        'efficientvit_b2': 3.7,
        'distilbert_base': 8.030415,
        'distilgpt2': 5.0,

        'prefill': 170.2,
        'decode': 20.7,
    }
    factor = [1, 2, 3, 4, 5, 6, 7, 8]
    slo = defaultdict(int)
    prefill_slo = defaultdict(int)
    decode_slo = defaultdict(int)

    # print(ltcs_at_tp)
    if not is_llm(ltcs_at_tp[0][0]):
        for model_name, _, _, ltc in ltcs_at_tp:
            if model_name in std_ltcs:
                for f in factor:
                    if ltc < std_ltcs[model_name] * f:
                        slo[f] += 1
            else:
                raise ValueError(f'Unknown model {model_name}')
        for k in slo:
            slo[k] /= len(ltcs_at_tp)
    else:
        for model_name, _, _, time_to_first_token_ltc in time_to_first_token_ltc_at_tp:
            for f in factor:
                if time_to_first_token_ltc < std_ltcs['prefill'] * f:
                    prefill_slo[f] += 1
        for k in prefill_slo:
            prefill_slo[k] /= len(time_to_first_token_ltc_at_tp)

        for model_name, _, _, time_between_token_ltc in time_between_token_ltc_at_tp:
            for f in factor:
                if time_between_token_ltc < std_ltcs['decode'] * f:
                    decode_slo[f] += 1
        for k in decode_slo:
            decode_slo[k] /= len(time_between_token_ltc_at_tp)

    if not is_llm(ltcs_at_tp[0][0]):
        fig, ax = plt.subplots()
        ax.bar([f'{k}x' for k in factor], [slo[k] for k in factor], width=0.5)
        ax.set_xlabel('Factor')
        ax.set_ylabel('SLO (%)')
        ax.set_ylim(0, 1)
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        for i in range(len(factor)):
            ax.text(i, 1.01, f'{slo[factor[i]]:.2f}', va='bottom', ha='center',
                    transform=trans)
        plt.savefig(slo_output_file)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1: plt.Axes
        ax2: plt.Axes
        
        ax1.bar([f'{k}x' for k in factor], [prefill_slo[k] for k in factor], width=0.5)
        ax1.set_xlabel('Factor')
        ax1.set_ylabel('Prefill SLO (%)')
        ax1.set_ylim(0, 1)
        trans1 = transforms.blended_transform_factory(ax1.transData, ax1.transAxes)
        for i in range(len(factor)):
            ax1.text(i, 1.01, f'{prefill_slo[factor[i]]:.2f}', va='bottom', ha='center',
                     transform=trans1)
        
        ax2.bar([f'{k}x' for k in factor], [decode_slo[k] for k in factor], width=0.5)
        ax2.set_xlabel('Factor')
        ax2.set_ylabel('Decode SLO (%)')
        ax2.set_ylim(0, 1)
        trans2 = transforms.blended_transform_factory(ax2.transData, ax2.transAxes)
        for i in range(len(factor)):
            ax2.text(i, 1.01, f'{decode_slo[factor[i]]:.2f}', va='bottom', ha='center',
                     transform=trans2)
        
        plt.tight_layout()
        plt.savefig(slo_output_file)


if output_file is not None:
    ltcs_at_tp = sorted(ltcs_at_tp, key=lambda x: x[1])

    with open(output_file, 'w') as F:
        F.write('model,start_timestamp,end_timestamp,ltc\n')
        for i in range(len(ltcs_at_tp)):
            F.write(f'{ltcs_at_tp[i][0]},{ltcs_at_tp[i][1]},{ltcs_at_tp[i][2]}, {ltcs_at_tp[i][3]}\n')