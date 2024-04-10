from __future__ import annotations
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import re


def eval_throughput(train_timeline: pd.DataFrame, infer_timeline: pd.DataFrame):
    if train_timeline.empty:
        return -1, -1, -1, -1, -1, -1, -1
    infer_begin = np.min(infer_timeline.start_time)
    infer_end = np.max(infer_timeline.end_time)
    col_mask = np.logical_and(train_timeline.start_time >= infer_begin, train_timeline.end_time <= infer_end)
    col_finished_mask = np.logical_and(col_mask, train_timeline.finished == True)
    col_cancel_mask = np.logical_and(col_mask, train_timeline.finished == False)
    num_image = np.sum(train_timeline.batch_size[col_finished_mask])
    elapsed_time = np.max(train_timeline.end_time[col_mask]) - np.min(train_timeline.start_time[col_mask])
    real_thpt = num_image / elapsed_time * 1000
    ideal_thpt = np.sum(train_timeline.batch_size[col_finished_mask]) / np.sum(train_timeline.end_time[col_finished_mask] - train_timeline.start_time[col_finished_mask]) * 1000
    avg_batch_time = np.mean(train_timeline.duration[col_finished_mask])
    avg_batch_size = np.mean(train_timeline.batch_size[col_finished_mask])
    num_finished_batch = np.sum(train_timeline.finished[col_mask])
    num_canceled_batch = np.sum(train_timeline.finished[col_mask] == False)
    finished_computing_time = np.sum(train_timeline.duration[col_finished_mask])
    canceled_computing_time = np.sum(train_timeline.duration[col_cancel_mask])
    waste_pct = canceled_computing_time / (finished_computing_time + canceled_computing_time) * 100
    return ideal_thpt, real_thpt, avg_batch_time, avg_batch_size, \
        num_finished_batch, num_canceled_batch, waste_pct

def eval_throughput_by_epoch(train_timeline: pd.DataFrame, infer_timeline: pd.DataFrame):
    if train_timeline.empty:
        return -1, -1, -1
    infer_begin = np.min(infer_timeline['start_time'])
    infer_end = np.max(infer_timeline['end_time'])
    colocated_train_timeline = train_timeline[train_timeline['start_time'] >= infer_begin]
    colocated_train_timeline = colocated_train_timeline[colocated_train_timeline['end_time'] <= infer_end]

    colocated_train_timeline['thpt'] = colocated_train_timeline['epoch_size'] / colocated_train_timeline['duration'] * 1000
    print('colocated epoch:\n', colocated_train_timeline, '\n')
    avg_thpt = np.mean(colocated_train_timeline['thpt'])

    tot_sample = np.sum(colocated_train_timeline['epoch_size'])
    e2e_thpt = tot_sample / (np.max(colocated_train_timeline['end_time']) - np.min(colocated_train_timeline['start_time'])) * 1000

    avg_duration = np.mean(colocated_train_timeline['duration']) / 1000

    return avg_thpt, e2e_thpt, avg_duration


def read_dataframe(log_dir: Path, train_timeline_name: str, infer_timeline_name: str, require: bool = False) -> Optional[tuple[pd.DataFrame, pd.DataFrame]]:
    train_timeline = log_dir / f"{train_timeline_name}"
    infer_timeline = log_dir / f"{infer_timeline_name}"
    if train_timeline.exists() and infer_timeline.exists():
        infer_timeline = pd.read_csv(infer_timeline)
        train_timeline = pd.read_csv(train_timeline)
        batch_name = r'batch_(\d+)_(\d+)_(\d+)' # epoch, batch, size
        train_batch_timeline = train_timeline[train_timeline['name'].str.match(batch_name)]
        batch_info = {}
        batch_info['timestamp'] = train_batch_timeline['timestamp']
        batch_info['epoch'] = train_batch_timeline['name'].str.extract(batch_name)[0].astype(int)
        batch_info['batch'] = train_batch_timeline['name'].str.extract(batch_name)[1].astype(int)
        batch_info['batch_size'] = train_batch_timeline['name'].str.extract(batch_name)[2].astype(int)
        batch_info['duration'] = train_batch_timeline['duration']
        batch_info['start_time'] = train_batch_timeline['timestamp']
        batch_info['end_time'] = train_batch_timeline['timestamp'] + train_batch_timeline['duration']
        batch_info['tag'] = train_batch_timeline['tag']
        batch_info['finished'] = train_batch_timeline['tag'] == 'finish'

        epoch_name = r'epoch_(\d+)_(\d+)' # epoch, size
        train_epoch_timeline = train_timeline[train_timeline['name'].str.match(epoch_name)]
        epoch_info = {}
        epoch_info['timestamp'] = train_epoch_timeline['timestamp']
        epoch_info['epoch'] = train_epoch_timeline['name'].str.extract(epoch_name)[0].astype(int)
        epoch_info['epoch_size'] = train_epoch_timeline['name'].str.extract(epoch_name)[1].astype(int)
        epoch_info['start_time'] = train_epoch_timeline['timestamp']
        epoch_info['duration'] = train_epoch_timeline['duration']
        epoch_info['end_time'] = train_epoch_timeline['timestamp'] + train_epoch_timeline['duration']

        global_batch_name = r'global_batch_(\d+)_(\d+)_(\d+)' # epoch, batch, size
        global_batch_timelime = train_timeline[train_timeline['name'].str.match(global_batch_name)]
        global_batch_info = {}
        global_batch_info['timestamp'] = global_batch_timelime['timestamp']
        global_batch_info['epoch'] = global_batch_timelime['name'].str.extract(global_batch_name)[0].astype(int)
        global_batch_info['batch'] = global_batch_timelime['name'].str.extract(global_batch_name)[1].astype(int)
        global_batch_info['batch_size'] = global_batch_timelime['name'].str.extract(global_batch_name)[2].astype(int)
        global_batch_info['duration'] = global_batch_timelime['duration']
        global_batch_info['start_time'] = global_batch_timelime['timestamp']
        global_batch_info['end_time'] = global_batch_timelime['timestamp'] + global_batch_timelime['duration']
        global_batch_info['tag'] = global_batch_timelime['tag']
        global_batch_info['finished'] = global_batch_timelime['tag'] == 'finish'

        return pd.DataFrame(epoch_info), pd.DataFrame(global_batch_info), pd.DataFrame(batch_info), infer_timeline
    elif not require:
        return None, None, None, None
    else:
        raise Exception(f"{train_timeline} or {infer_timeline} does not exist.")
    

def main():
    parser = ArgumentParser()
    parser.add_argument("--log-dir", type=Path, help="log dir", required=True)
    parser.add_argument("--train-timeline", type=str, help="train timeline csv file path", default="train-profile.csv")
    parser.add_argument("--infer-timeline", type=str, help="train timeline csv file path", default="infer-timeline")
    args = parser.parse_args()
    log_dir: Path = args.log_dir
    train_timeline_name = args.train_timeline
    infer_timeline_name = args.infer_timeline
    epoch_timeline, global_batch_timelime, batch_timeline, infer_timeline = read_dataframe(log_dir, train_timeline_name, infer_timeline_name)
    if infer_timeline is None:
        for sub_log_dir in sorted(filter(lambda file_or_dir: file_or_dir.is_dir(), log_dir.iterdir()), key=lambda must_dir: str(must_dir)):
            try:
                epoch_timeline, global_batch_timelime, batch_timeline, infer_timeline = read_dataframe(sub_log_dir, train_timeline_name, infer_timeline_name, require=True)
                ideal_thpt, real_thpt, avg_batch_time, avg_batch_size, num_finished_batch, num_canceled_batch, waste_pct = eval_throughput(batch_timeline, infer_timeline)
                print(f"[{sub_log_dir.name}] thpt: {real_thpt:.2f} / {ideal_thpt:.2f} it/sec, avg_batch_time {avg_batch_time:.1f}, avg_batch_size {avg_batch_size:.1f}, \
num_finished {num_finished_batch} num_cancel {num_canceled_batch} waste_pct {waste_pct:.1f}\n")
            except Exception as e:
                print(f"[{sub_log_dir.name}] {e}\n")
    else:
        ideal_thpt, real_thpt, avg_batch_time, avg_batch_size, num_finished_batch, num_canceled_batch, waste_pct = eval_throughput(batch_timeline, infer_timeline)
        print(f"[{log_dir.name}] thpt: {real_thpt:.2f} / {ideal_thpt:.2f} it/sec, avg_batch_time {avg_batch_time:.1f}, avg_batch_size {avg_batch_size:.1f}, \
num_finished {num_finished_batch} num_cancel {num_canceled_batch}, waste_pct {waste_pct:.1f}\n")
        
        gb_ideal_thpt, gb_real_thpt, gb_avg_batch_time, gb_avg_batch_size, gb_num_finished_batch, gb_num_canceled_batch, gb_waste_pct = eval_throughput(global_batch_timelime, infer_timeline)
        print(f"[{log_dir.name}] global batch thpt: {gb_real_thpt:.2f} / {gb_ideal_thpt:.2f} it/sec, avg_batch_time {gb_avg_batch_time:.1f}, avg_batch_size {gb_avg_batch_size:.1f}, \
num_finished {gb_num_finished_batch} num_cancel {gb_num_canceled_batch}, waste_pct {gb_waste_pct:.1f}\n")

        epoch_avg_thpt, epoch_e2e_thpt, epoch_avg_duration = eval_throughput_by_epoch(epoch_timeline, infer_timeline)
        print(f'[{log_dir.name}] epoch e2e thpt: {epoch_e2e_thpt:.2f} it/sec, avg thpt (over epoch): {epoch_avg_thpt:.2f} it/sec, \
avg duration {epoch_avg_duration:.2f}\n')
    
if __name__ == "__main__":
    main()