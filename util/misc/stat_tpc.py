import numpy as np
import pandas as pd
import os, sys, re
import argparse

from collections import defaultdict

def parse_log(log):
    prompt, decode = defaultdict(list), defaultdict(list)
    with open(log) as f:
        for line in f.readlines():
            if 'TPC_PERF:' in line:
                # Extract values using regex
                match = re.search(r'TPC_PERF: (\d+) (\d+) (\d+) (\d+\.\d+)', line)
                if match:
                    is_prompt = int(match.group(1))
                    num_token = int(match.group(2))
                    num_tpc = int(match.group(3))
                    exec_time = float(match.group(4))
                    # data.append([is_prompt, num_token, num_tpc, exec_time])
                    if is_prompt:
                        prompt[num_token].append(exec_time)
                    else:
                        decode[num_token].append(exec_time)
    # df = pd.DataFrame(data, columns=['is_prompt', 'num_token', 'num_tpc', 'exec_time'])
    return (
        {k: np.mean(v) for k, v in prompt.items()},
        {k: np.mean(v) for k, v in decode.items()}
    )

parser = argparse.ArgumentParser()
parser.add_argument('--oracle', type=str, help='Path to server log file')
parser.add_argument('--log', type=str, help='Path to server log file')

args = parser.parse_args()

oracle_prompt, oracle_decode = parse_log(args.oracle)
tested_prompt, tested_decode = parse_log(args.log)

# print(tested_prompt)

# print(oracle_prompt)


prompt_token_bin = [
    128, 256, 512, 1024, 2048, 4096
]

decode_token_bin = [
    1, 2, 4, 8, 16, 32, 64, 128
]

for slow_down in [
    1.1, 1.2, 1.3, 1.4, 1.5,
    2, 10,
]:
    print(f'\n==== slow down {slow_down} ====')

    prompt_slo = defaultdict(int)
    prompt_tot = defaultdict(int)
    decode_slo = defaultdict(int)
    decode_tot = defaultdict(int)

    # prompt
    for token, exec_time in tested_prompt.items():
        if token not in oracle_prompt:
            # print(f'prompt {token} not find in oracle')
            # continue
            continue

        bin = 0
        while bin < len(prompt_token_bin) and token > prompt_token_bin[bin]:
            bin += 1
        
        bin = prompt_token_bin[bin]
        slo = oracle_prompt[token] * slow_down

        # print(exec_time, slo)
        if exec_time < slo:
            prompt_slo[bin] += 1
        prompt_tot[bin] += 1

    print('prompt: ', end='')
    for b in prompt_token_bin:
        print(f'{b} {prompt_slo[b]}/{prompt_tot[b]}', end=' | ')

    # decode
    for token, exec_time in tested_decode.items():
        bin = 0
        while bin < len(decode_token_bin) and token > decode_token_bin[bin]:
            bin += 1
        if token not in oracle_decode:
            # print(f'decode {token} not find in oracle')
            continue

        bin = decode_token_bin[bin]     
        slo = oracle_decode[token] * slow_down
        if exec_time < slo:
            decode_slo[bin] += 1
        decode_tot[bin] += 1

    print('\ndecode: ', end='')
    for b in decode_token_bin:
        print(f'{b} {decode_slo[b]}/{decode_tot[b]}', end=' | ')




