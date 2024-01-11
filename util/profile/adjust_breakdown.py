import numpy as np
import pandas as pd
import argparse 

app = argparse.ArgumentParser("adjust breakdown")
app.add_argument("-l", type=str)
args = app.parse_args()

train_profile = args.l

df = pd.read_csv(train_profile, index_col=0)
df = df.sort_index()
# df['name_id'] = df['name'].apply(lambda x: 1 if x == 'adjust_l1' or x == 'adjust_l2' else 0)
# df = df.sort_values(['timestamp', 'name_id'], ascending=[True, True])

wait_adjust_time = []
adjust_release_time = []

recv_adjust_time = None
for time_stamp, row in df.iterrows():
    # time_stamp = row['timestamp']
    if row['name'] == 'recv_adjust' and recv_adjust_time is None:
        recv_adjust_time = time_stamp
    elif (row['name'] == 'adjust_l1' or row['name'] == 'adjust_l2'):
        assert recv_adjust_time is not None, f"recv_adjust_time is None at {time_stamp}"
        wait_adjust_time.append(time_stamp - recv_adjust_time)
        # if time_stamp - recv_adjust_time > 200:
        #     print(time_stamp, recv_adjust_time)
        adjust_release_time.append(int(row['duration']))
    if row['name'] == 'adjust_done':
        assert recv_adjust_time is not None, f"recv_adjust_time is None at {time_stamp}"
        recv_adjust_time = None

print("wait_adjust: mean {:.1f} std {:.1f} max {} cnt {:.1f}".format(
    np.mean(wait_adjust_time), np.std(wait_adjust_time), np.max(wait_adjust_time), len(wait_adjust_time)))
print("adjust_release: mean {:.1f} std {:.1f} cnt {:.1f}".format(
    np.mean(adjust_release_time), np.std(adjust_release_time), len(adjust_release_time)))
    

