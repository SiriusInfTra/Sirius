import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--trace', type=str, required=True)
parser.add_argument('-o', '--outdir', type=str, required=True)
args = parser.parse_args()

trace = args.trace
outdir = args.outdir

trace_list = []
with open(trace, 'r') as f:
    lines = f.readlines()
    start = False
    for l in lines:
        if not start:
            if l.strip() != '# start_point,model_id':
                continue
            else:
                start = True
        else:
            start_point, model_id = l.strip().split(',')
            trace_list.append((float(start_point), model_id))

start_points = [x[0] for x in trace_list]

# print(start_points)

fig,ax = plt.subplots()
ax.hist(start_points, bins=range(0, int(max(start_points)+0.5), 1))

plt.savefig(outdir + '/trace.svg')
