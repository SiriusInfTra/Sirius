import argparse
import re
import pandas

parser = argparse.ArgumentParser('Collect inference latency')
parser.add_argument('-l', '--log', type=str, required=True)
parser.add_argument('-o', '--output', type=str, required=True)

args = parser.parse_args()

log_file = args.log
output_file = args.output


parse_ltc = False
ltcs_at_tp = []
with open(log_file, 'r') as F:
    for line in F.readlines():
        if line.strip() == '':
            parse_ltc = False
            continue
        if 'InferWorker TRACE' in line:
            parse_ltc = True
            continue
        if '=' in line or '[ ' in line or ']' in line:
            parse_ltc = False
            continue

        if parse_ltc: 
            data = line.strip().split(',')
            start_tp = int(data[0].strip())
            end_tp = int(data[1].strip())
            ltc = float(data[-1].strip())
            ltcs_at_tp.append((start_tp, end_tp, ltc))
            # start_tps.append(start_tp)
            # ltcs.append(ltc)
        
ltcs_at_tp = sorted(ltcs_at_tp, key=lambda x: x[0])

with open(output_file, 'w') as F:
    F.write('start_timestamp,end_timestamp,ltc\n')
    for i in range(len(ltcs_at_tp)):
        F.write(f'{ltcs_at_tp[i][0]},{ltcs_at_tp[i][1]},{ltcs_at_tp[i][2]}\n')