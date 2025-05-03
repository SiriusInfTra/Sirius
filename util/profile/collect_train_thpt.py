import argparse
import re
import pandas as pd

parser = argparse.ArgumentParser('Collect inference latency')
parser.add_argument('-l', '--log', type=str, required=True)
parser.add_argument('-o', '--output', type=str, required=True)
args = parser.parse_args()

train_timeline = args.log

train_timeline = pd.read_csv(train_timeline)

batch_name = r'batch_(\d+)_(\d+)_(\d+)' # epoch, batch, size
train_batch_timeline = train_timeline[train_timeline['name'].str.match(batch_name)]
batch_info = {}
batch_info['timestamp'] = train_batch_timeline['timestamp'].tolist()
batch_info['epoch'] = train_batch_timeline['name'].str.extract(batch_name)[0].astype(int)
batch_info['batch'] = train_batch_timeline['name'].str.extract(batch_name)[1].astype(int)
batch_info['batch_size'] = train_batch_timeline['name'].str.extract(batch_name)[2].astype(int).tolist()
batch_info['duration'] = train_batch_timeline['duration']
batch_info['start_time'] = train_batch_timeline['timestamp'].tolist()
batch_info['end_time'] = (train_batch_timeline['timestamp'] + train_batch_timeline['duration']).tolist()
batch_info['tag'] = train_batch_timeline['tag']
batch_info['finished'] = (train_batch_timeline['tag'] == 'finish').tolist()


with open(args.output, 'w') as f:
    f.write('start_timestamp,end_timestamp,batch_size\n')
    for i in range(len(batch_info['timestamp'])):
        # print('##', i)
        # print(batch_info['start_time'][i])
        # print(batch_info['end_time'][i])
        # print(batch_info['batch_size'][i])
        if batch_info['finished'][i]:
            f.write('{},{},{}\n'.format(batch_info['start_time'][i], batch_info['end_time'][i], batch_info['batch_size'][i]))


