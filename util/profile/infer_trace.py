import sys
import re
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import argparse

app = argparse.ArgumentParser("infer trace")
app.add_argument('-l', '--log', type=str, required=True, )
app.add_argument('-o', '--outdir', type=str, required=False, default=None)
app.add_argument('-g', '--group-model', action='store_true')
app.add_argument('--drop-first', type=int, default=0)
app.add_argument('-v', '--verbose', action='store_true')
args = app.parse_args()

log = args.log
outdir = args.outdir
group = args.group_model
drop_first = args.drop_first
infers = []

if outdir is not None:
    if not pathlib.Path(outdir).exists():
        pathlib.Path(outdir).mkdir()

with open(log, "r") as F:
    while line := F.readline():
        if m := re.search(r"\[InferWorker TRACE (([0-9a-zA-Z]+)(-\d+)?)\]", line):
            if not group:
                cur_model = m.group(1)
            else:
                cur_model = m.group(2)
            
            drop_cnt = 0
            while line := F.readline():
                if m := re.search(r"([0-9]+\.[0-9]+), ([0-9]+\.[0-9]+), ([0-9]+\.[0-9]+)", line):
                    if drop_cnt < drop_first:
                        drop_cnt += 1
                        continue
                    infers.append((
                        cur_model, 
                        float(m.group(1)), 
                        float(m.group(2)), 
                        float(m.group(3))
                    ))
                else:
                    break

infers = pd.DataFrame(infers, columns=["model", "request_time", "response_time", "latency"])

if args.verbose:
    print(infers)

for model in set(infers["model"]):
    model_infers = infers[infers["model"] == model]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    reqeust_time = model_infers["request_time"]
    response_time = model_infers["response_time"]
    latency = model_infers["latency"]

    if group:
        resp_ltc = [(resp, ltc) for resp, ltc in zip(reqeust_time, latency)]
        resp_ltc = sorted(resp_ltc, key=lambda x: x[0])
        reqeust_time, latency = zip(*resp_ltc)

    axs[0].hist(reqeust_time, range(0, int(max(response_time)) // 1000 * 1000, 1000))
    axs[0].set_title(f"{model} request time hist")
    axs[1].plot(reqeust_time, latency)
    axs[1].set_title(f"{model} latency")
    sorted_latecy = sorted(latency)
    axs[2].plot([i / len(latency) for i in range(len(latency))], sorted_latecy)
    axs[2].hlines(sorted_latecy[int(len(sorted_latecy) * 0.99)], 0, 1, colors="r", linestyles="dashed")
    axs[2].set_title(f"{model} latency cdf")

    if outdir is not None:
        plt.savefig(f"{outdir}/{model}.svg")
        print(f"save {model}.svg")
    
    print('''{}:
    cnt: {:7} | max: {:7} | min: {:7}
    p99: {:7} | p95: {:7} | p90: {:7}
'''.format(model,
           len(sorted_latecy), max(sorted_latecy), min(sorted_latecy),
           sorted_latecy[int(len(sorted_latecy) * 0.99)],
           sorted_latecy[int(len(sorted_latecy) * 0.95)],
           sorted_latecy[int(len(sorted_latecy) * 0.90)]))

        
        