import subprocess
import os, re
import numpy as np

def calcu_avg_slo(live_time, hit_rate):

    cmd = "python util/modeling/matrix_method.py --rps {} "
    cmd += "--service-time 0.0073 --slo 0.0292 --cold-start-time 0.0098 "
    cmd += f"--live-time {live_time} --hite-rate {hit_rate} | tail -n 1"

    slos = []
    for rps in [98.2, 29.9, 29.9, 3.5, 20.7]:
        cmd_ = cmd.format(rps)
        proc = subprocess.run(cmd_, shell=True, capture_output=True, text=True)
        slo = re.search(r'0\.\d+', proc.stdout)
        if slo:
            slo = float(slo.group(0))
            slos.append(slo)
        else:
            print(f"Failed to parse SLO for RPS {rps}")

    return np.mean(slos), slos


for live_time in [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    for hit_rate in [0.09385311 * i for i in range(5)]:
        slo, slos = calcu_avg_slo(live_time, hit_rate)
        print(f"Live time: {live_time}, Hit rate: {hit_rate}, SLO: {slo} | {' '.join([f'{s:.2f}' for s in slos])}")

