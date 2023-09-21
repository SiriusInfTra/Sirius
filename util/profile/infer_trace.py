import sys
import re
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import argparse

app = argparse.ArgumentParser("infer trace")
app.add_argument('-l', type=str, required=True)
app.add_argument('-o', type=str, required=True)
app.add_argument('-g', action='store_true')
args = app.parse_args()


log = args.l
outdir = args.o
group = args.g
infers = []

if not pathlib.Path(outdir).exists():
    pathlib.Path(outdir).mkdir()

with open(log, "r") as F:
    cur_model = None
    while line := F.readline():
        if not cur_model:
          if m := re.search(r"\[InferWorker TRACE (([0-9a-zA-Z]+)(-\d+)?)\]", line):
            if not group:
                cur_model = m.group(1)
            else:
                cur_model = m.group(2)
        else:
            if m := re.search(r"([0-9]+\.[0-9]+), ([0-9]+\.[0-9]+), ([0-9]+\.[0-9]+)", line):
                infers.append((
                   cur_model, 
                   float(m.group(1)), 
                   float(m.group(2)), 
                   float(m.group(3))
                ))
            else:
                cur_model = None

infers = pd.DataFrame(infers, columns=["model", "request_time", "response_time", "latency"])


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

    axs[0].hist(reqeust_time, range(0, int(max(response_time)) // 1000 * 1000, 100))
    axs[0].set_title(f"{model} request time hist")
    axs[1].plot(reqeust_time, latency)
    axs[1].set_title(f"{model} latency")
    sorted_latecy = sorted(latency)
    axs[2].plot([i / len(latency) for i in range(len(latency))], sorted_latecy)
    axs[2].hlines(sorted_latecy[int(len(sorted_latecy) * 0.99)], 0, 1, colors="r", linestyles="dashed")
    axs[2].set_title(f"{model} latency cdf")

    plt.savefig(f"{outdir}/{model}.svg")
    print(f"save {model}.svg")
    

        
        