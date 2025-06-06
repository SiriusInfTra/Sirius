from __future__ import annotations
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
import re


def eval_throughput(train_timeline: pd.DataFrame, 
                    infer_timeline: pd.DataFrame,
                    restart_timeline: Optional[pd.DataFrame],
                    wait_valid_batch_timeline: Optional[pd.DataFrame]):
    if train_timeline.empty:
        return -1, -1, -1, -1, -1, -1, -1, -1
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
    # waste_pct = canceled_computing_time / (finished_computing_time + canceled_computing_time) * 100
    waste_pct = canceled_computing_time / elapsed_time * 100

    if restart_timeline is not None and not restart_timeline.empty:
        colocated_restart_timeline: pd.DataFrame = restart_timeline[restart_timeline['start_time'] >= infer_begin]
        colocated_restart_timeline = colocated_restart_timeline[colocated_restart_timeline['end_time'] <= infer_end]
        restart_waste = np.sum(colocated_restart_timeline['duration'])
        restart_waste_pct = restart_waste / elapsed_time * 100
    else:
        restart_waste = -1
        restart_waste_pct = -1

    if wait_valid_batch_timeline is not None and not wait_valid_batch_timeline.empty:
        col_wait_valid_batch_timeline = wait_valid_batch_timeline[wait_valid_batch_timeline['start_time'] >= infer_begin]
        col_wait_valid_batch_timeline = col_wait_valid_batch_timeline[col_wait_valid_batch_timeline['end_time'] <= infer_end]
        wait_valid_waste = np.sum(col_wait_valid_batch_timeline['duration'])
        wait_valid_waste_pct = wait_valid_waste / elapsed_time * 100
    else:
        wait_valid_waste = -1
        wait_valid_waste_pct = -1

    print(f"[eval_throughput] elapsed_time {elapsed_time:.2f} "
          f"compute {finished_computing_time:.2f} ({finished_computing_time / elapsed_time * 100:.2f}%) "
          f"cancel waste {canceled_computing_time:.2f} ({canceled_computing_time / elapsed_time * 100:.2f}%) "
          f"restart wast {restart_waste:.2f} ({restart_waste_pct:.2f}%) "
          f"wait valid waste {wait_valid_waste:.2f} ({wait_valid_waste_pct:.2f}%) ")

    return (ideal_thpt, real_thpt, avg_batch_time, avg_batch_size, 
        num_finished_batch, num_canceled_batch, waste_pct, restart_waste_pct)


def eval_throughput_by_epoch(train_timeline: pd.DataFrame, infer_timeline: pd.DataFrame):
    if train_timeline.empty:
        return -1, -1, -1
    infer_begin = np.min(infer_timeline['start_time'])
    infer_end = np.max(infer_timeline['end_time'])
    colocated_train_timeline = train_timeline[train_timeline['start_time'] >= infer_begin]
    colocated_train_timeline = colocated_train_timeline[colocated_train_timeline['end_time'] <= infer_end]

    colocated_train_timeline['thpt'] = colocated_train_timeline['epoch_size'] / colocated_train_timeline['duration'] * 1000
    pd.options.display.float_format = '{:.2f}'.format
    print('colocated epoch:\n', colocated_train_timeline.to_string(), '\n')
    avg_thpt = np.mean(colocated_train_timeline['thpt'])

    tot_sample = np.sum(colocated_train_timeline['epoch_size'])
    e2e_thpt = tot_sample / (np.max(colocated_train_timeline['end_time']) - np.min(colocated_train_timeline['start_time'])) * 1000

    avg_duration = np.mean(colocated_train_timeline['duration']) / 1000

    return avg_thpt, e2e_thpt, avg_duration


def read_dataframe(log_dir: Path, 
                   train_timeline_name: str, 
                   infer_timeline_name: str) -> List[List[Optional[pd.DataFrame]]]:
    train_timeline = log_dir / f"{train_timeline_name}"
    infer_timeline = log_dir / f"{infer_timeline_name}"

    if not (train_timeline.exists() and infer_timeline.exists()):
        raise Exception(f"{train_timeline} or {infer_timeline} does not exist.")

    infer_timeline = pd.read_csv(infer_timeline)
    train_timeline = pd.read_csv(train_timeline)

    ret_dfs = []
    print(train_timeline.columns)
    train_world_size = train_timeline['rank'].nunique()

    for rank in range(train_world_size):
        rank_train_timeline = train_timeline[train_timeline['rank'] == rank]

        batch_name = r'batch_(\d+)_(\d+)_(\d+)' # epoch, batch, size
        train_batch_timeline = rank_train_timeline[rank_train_timeline['name'].str.match(batch_name)]
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

        batch_restart_name = r'batch_restart_(\d+)_(\d+)' # epoch, batch
        train_batch_restart_timeline = \
            rank_train_timeline[rank_train_timeline['name'].str.match(batch_restart_name)]
        batch_restart_info = {}
        batch_restart_info['timestamp'] = train_batch_restart_timeline['timestamp']
        batch_restart_info['epoch'] = \
            train_batch_restart_timeline['name'].str.extract(batch_restart_name)[0].astype(int)
        batch_restart_info['batch'] = \
            train_batch_restart_timeline['name'].str.extract(batch_restart_name)[1].astype(int)
        batch_restart_info['start_time'] = train_batch_restart_timeline['timestamp']
        batch_restart_info['duration'] = train_batch_restart_timeline['duration']
        batch_restart_info['end_time'] = \
            train_batch_restart_timeline['timestamp'] + train_batch_restart_timeline['duration']
        
        wait_valid_batch_name = r'wait_valid_batch'
        train_wait_valid_batch_timeline = \
            rank_train_timeline[rank_train_timeline['name'].str.match(wait_valid_batch_name)]
        wait_valid_batch_info = {}
        wait_valid_batch_info['timestamp'] = train_wait_valid_batch_timeline['timestamp']
        wait_valid_batch_info['start_time'] = train_wait_valid_batch_timeline['timestamp']
        wait_valid_batch_info['duration'] = train_wait_valid_batch_timeline['duration']
        wait_valid_batch_info['end_time'] = \
            train_wait_valid_batch_timeline['timestamp'] + train_wait_valid_batch_timeline['duration']

        epoch_name = r'epoch_(\d+)_(\d+)' # epoch, size
        train_epoch_timeline = rank_train_timeline[rank_train_timeline['name'].str.match(epoch_name)]
        epoch_info = {}
        epoch_info['timestamp'] = train_epoch_timeline['timestamp']
        epoch_info['epoch'] = train_epoch_timeline['name'].str.extract(epoch_name)[0].astype(int)
        epoch_info['epoch_size'] = train_epoch_timeline['name'].str.extract(epoch_name)[1].astype(int)
        epoch_info['start_time'] = train_epoch_timeline['timestamp']
        epoch_info['duration'] = train_epoch_timeline['duration']
        epoch_info['end_time'] = train_epoch_timeline['timestamp'] + train_epoch_timeline['duration']

        global_batch_name = r'global_batch_(\d+)_(\d+)_(\d+)' # epoch, batch, size
        global_batch_timelime = rank_train_timeline[rank_train_timeline['name'].str.match(global_batch_name)]
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

        ret_dfs.append((pd.DataFrame(epoch_info), 
                        pd.DataFrame(global_batch_info), 
                        pd.DataFrame(batch_info), 
                        pd.DataFrame(batch_restart_info), 
                        pd.DataFrame(wait_valid_batch_info),
                        infer_timeline))
    return ret_dfs

    

def main():
    parser = ArgumentParser()
    parser.add_argument("--log-dir", type=Path, help="log dir", required=True)
    parser.add_argument("--train-timeline", type=str, help="train timeline csv file path", default="train-profile.csv")
    parser.add_argument("--infer-timeline", type=str, help="train timeline csv file path", default="infer-timeline")
    args = parser.parse_args()
    log_dir: Path = args.log_dir
    train_timeline_name = args.train_timeline
    infer_timeline_name = args.infer_timeline
    # epoch_timeline, global_batch_timelime, batch_timeline, infer_timeline = read_dataframe(log_dir, train_timeline_name, infer_timeline_name)
    train_profile_dfs = read_dataframe(log_dir, train_timeline_name, infer_timeline_name)

    if not train_profile_dfs:
        raise Exception(f"get empty train profile dataframes")

    for rank, (epoch_timeline, 
               global_batch_timelime, 
               batch_timeline, 
               batch_restart_timeline,
               wait_valid_batch_timeline,
               infer_timeline) in enumerate(train_profile_dfs):
        (
            ideal_thpt, 
            real_thpt, 
            avg_batch_time, 
            avg_batch_size, 
            num_finished_batch, 
            num_canceled_batch, 
            waste_pct,
            restart_waste_pct
        ) = eval_throughput(batch_timeline, infer_timeline, 
                            batch_restart_timeline, wait_valid_batch_timeline)

        print(f"[Rank {rank} | {log_dir.name}] "
              f"thpt: {real_thpt:.2f} / {ideal_thpt:.2f} it/sec, "
              f"avg_batch_time {avg_batch_time:.1f}, "
              f"avg_batch_size {avg_batch_size:.1f}, "
              f"num_finished {num_finished_batch} num_cancel {num_canceled_batch}, "
              f"waste_pct {waste_pct:.1f}, "
              f"restart_waste_pct {restart_waste_pct:.1f}, "
               "\n")
        
        (
            gb_ideal_thpt, 
            gb_real_thpt, 
            gb_avg_batch_time, 
            gb_avg_batch_size, 
            gb_num_finished_batch, 
            gb_num_canceled_batch, 
            gb_waste_pct,
            _
        ) = eval_throughput(global_batch_timelime, infer_timeline, None, None)
        print(f"[Rank {rank} | {log_dir.name}] "
              f"global batch thpt: {gb_real_thpt:.2f} / {gb_ideal_thpt:.2f} it/sec, "
              f"avg_batch_time {gb_avg_batch_time:.1f}, "
              f"avg_batch_size {gb_avg_batch_size:.1f}, "
              f"num_finished {gb_num_finished_batch} num_cancel {gb_num_canceled_batch}, "
              f"waste_pct {gb_waste_pct:.1f}\n")

        (
            epoch_avg_thpt, 
            epoch_e2e_thpt, 
            epoch_avg_duration
        ) = eval_throughput_by_epoch(epoch_timeline, infer_timeline)
        print(f'[Rank {rank} | {log_dir.name}] '
              f"epoch e2e thpt: {epoch_e2e_thpt:.2f} it/sec, "
              f"avg thpt (over epoch): {epoch_avg_thpt:.2f} it/sec, "
              f"avg duration {epoch_avg_duration:.2f}\n")
        
        if rank + 1 != len(train_profile_dfs):
            print(f"==============================\n\n")

    
if __name__ == "__main__":
    main()