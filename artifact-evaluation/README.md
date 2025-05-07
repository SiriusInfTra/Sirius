## Overview

<style type="text/css">
u {    
    border-bottom: 1px dashed #000;
    text-decoration: none;
}
</style>

A bash script ([./eval/run_all.sh](../eval/run_all.sh)) is provided to wrap evaluation scripts, automating the evaluation process and data collection. To run all evaluations, simply execute `bash ./eval/run_all.sh`. The details of each evaluation are described below.

|Figure|Hardware|Command To Execute|Execution Time|
|-|-|-|-|
|[**Figure 9**](#figure-9)|1*`V100`|`bash ./eval/run_all.sh --overall-single-gpu`|4H:30M|
|[**Figure 10**](#figure-10)|4*`V100`|`bash ./eval/run_all.sh --overall-multi-gpu`|1H:20M|
|[**Figure 11**](#figure-11)|4*`V100`|`bash ./eval/run_all.sh --breakdown`|40M|
|[**Figure 12**](#figure-12)|1*`V100`|`bash ./eval/run_all.sh --ablation`|1H:30M|
|[**Figure 13**](#figure-13)|2*`V100`|`bash ./eval/run_all.sh --unbalance`|20M|
|[**Figure 14**](#figure-14)|1*`V100`|`bash ./eval/run_all.sh --memory-pressure`|30M|
|[**Figure 15**](#figure-15)|1*`A100`|`bash ./eval/run_all.sh --llm`|40M|

<br>

---

The following sections detail the evaluation commands executed by the bash script.

### Figure 9

```bash
source ./scripts/set_cuda_device.sh 0
python eval/overall_v2.py --uniform-v2 --skewed-v2 --azure \
    --sirius --static-partition --task-switch --um-mps --infer-only \
    --skip-set-mps-pct --retry-limit 3 --skip-fail 1 --parse-result
```

The command above evaluates Sirius and baselines with all workloads on a single GPU. After the evaluation completes, the results can be found in the log directory printed to the terminal, for example:

<pre>
...
Executing plot script: eval/runner/plot_scripts/overall_v100.py at <u>log/overall-uniform-v2-1gpu-YYYYMMDD-HHMM</u>
</pre>

### Figure 10

```bash
source ./scripts/set_cuda_device.sh 0 1 2 3
python eval/overall_v2.py --uniform-v2 --uniform-v2-wkld-types NormalLight \
    --sirius --static-partition --task-switch --um-mps --infer-only \
    --skip-set-mps-pct --multi-gpu --retry-limit 3 --skip-fail 1 --parse-result
```

The command above evaluates Sirius and baselines with the **Light** workload on 4 GPUs. Similarly, the results can be found in the directory printed to the terminal, for example:

<pre>
...
Executing plot script: eval/runner/plot_scripts/multi_gpu.py at <u>log/overall-uniform-v2-4gpu-YYYYMMDD-HHMM</u>
</pre>

### Figure 11

```bash
source ./scripts/set_cuda_device.sh 0
python eval/breakdown.py --sirius --strawman --azure --retry-limit 3 --parse-result

source ./scripts/set_cuda_device.sh 0 1 2 3
python eval/breakdown.py --sirius --strawman --azure --multi-gpu --retry-limit 3 --parse-result
```

The commands above break down the performance of memory handover on 1 GPU and 4 GPUs, respectively. The results can be found in the directories printed to the terminal, for example:

<pre>
...
Executing plot script: eval/runner/plot_scripts/adjust_breakdown_single_or_multi_gpu.py --single-gpu at <u>log/breakdown-azure-1gpu-YYYYMMDD-HHMM</u>

...
Executing plot script: eval/runner/plot_scripts/adjust_breakdown_single_or_multi_gpu.py --multi-gpu at <u>log/breakdown-azure-4gpu-YYYYMMDD-HHMM</u>
</pre>

### Figure 12

```bash
source ./scripts/set_cuda_device.sh 0
python eval/ablation.py --eval-all --retry-limit 3 --parse-result
```

The command above studies the impact of watermark and model idle time on a single GPU. The results can be found in the directory printed to the terminal, for example:

<pre>
...
Executing plot script: eval/runner/plot_scripts/ablation.py at <u>log/ablation-infer-idle-time-uniform-YYYYMMDD-HHMM</u>
</pre>

### Figure 13

```bash
source ./scripts/set_cuda_device.sh 0 1
python eval/unbalance.py --parse-result
```

The command above evaluates Sirius when memory requirements are unbalanced across 2 GPUs. The results can be found in the directory printed to the terminal, for example:

<pre>
...
Executing plot script: eval/runner/plot_scripts/memory_util_gpus.py at <u>log/overall-uniform-v2-2gpu-56-YYYYMMDD-HHMM</u>
</pre>


### Figure 14

```bash
source ./scripts/set_cuda_device.sh 0
python eval/memory_pressure.py --retry-limit 3 --parse-result
```

The command above evaluates Sirius under memory pressure on a single GPU. The results can be found in the directory printed to the terminal, for example:

<pre>
...
Executing plot script: eval/runner/plot_scripts/memory_pressure.py at <u>log/memory-pressure-YYYYMMDD-HHMM</u>
</pre>


### Figure 15

```bash
source ./scripts/set_cuda_device.sh 0
python eval/run_llm.py --sirius --static-partition --infer-only \
    --burstgpt --burstgpt-rps 10 --parse-result
```

The command above evaluates Sirius and Static Partition using the **BurstGPT** workload on a single GPU. The results can be found in the directory printed to the terminal, for example:

<pre>
...
Executing plot script: eval/runner/plot_scripts/llm.py at <u>log/burstgpt-1gpu-YYYYMMDD-HHMM</u>
</pre>