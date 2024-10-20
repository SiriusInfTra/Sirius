import argparse
import re
import pandas
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import transforms
import pathlib

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
with open(log_file, 'r') as F:
    for line in F.readlines():
        if line.strip() == '':
            parse_ltc = False
            continue
        # if 'InferWorker TRACE' in line:
        mat = re.search(r'\[InferWorker TRACE ([0-9a-zA-Z]+)-(\d+)\]', line)
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
            ltc = float(data[-1].strip())
            ltcs_at_tp.append((model_name, start_tp, end_tp, ltc))
            # start_tps.append(start_tp)
            # ltcs.append(ltc)

if slo_output_file is not None:
    std_ltcs = {
        'resnet152': 9.358644,
        'densenet161': 12.586594,
        'efficientnet': 5.287647,
        'efficientvit': 3.7,
        'distilbert_base': 8.030415,
        'distilgpt2': 5.0,
    }
    factor = [1, 2, 3, 4, 5, 6, 7, 8]
    slo = defaultdict(int)
    for model_name, _, _, ltc in ltcs_at_tp:
        if model_name in std_ltcs:
            for f in factor:
                if ltc < std_ltcs[model_name] * f:
                    slo[f] += 1
        else:
            raise ValueError(f'Unknown model {model_name}')
    for k in slo:
        slo[k] /= len(ltcs_at_tp)


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


if output_file is not None:
    ltcs_at_tp = sorted(ltcs_at_tp, key=lambda x: x[1])

    with open(output_file, 'w') as F:
        F.write('model,start_timestamp,end_timestamp,ltc\n')
        for i in range(len(ltcs_at_tp)):
            F.write(f'{ltcs_at_tp[i][0]},{ltcs_at_tp[i][1]},{ltcs_at_tp[i][2]}, {ltcs_at_tp[i][3]}\n')