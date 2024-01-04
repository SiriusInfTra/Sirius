from __future__ import annotations
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import re


def eval_throughput(train_timeline: pd.DataFrame, infer_timeline: pd.DataFrame):
    infer_begin = np.min(infer_timeline.start_time)
    infer_end = np.max(infer_timeline.end_time)
    col_mask = np.logical_and(train_timeline.start_time >= infer_begin, train_timeline.end_time <= infer_end)
    col_finished_mask = np.logical_and(col_mask, train_timeline.finished == True)
    num_image = np.sum(train_timeline.batch_size[col_finished_mask])
    elapsed_time = np.max(train_timeline.end_time[col_mask]) - np.min(train_timeline.start_time[col_mask])
    real_thpt = num_image / elapsed_time * 1000
    ideal_thpt = np.sum(train_timeline.batch_size[col_finished_mask]) / np.sum(train_timeline.end_time[col_finished_mask] - train_timeline.start_time[col_finished_mask]) * 1000
    avg_batch_time = np.mean(train_timeline.duration[col_finished_mask])
    avg_batch_size = np.mean(train_timeline.batch_size[col_finished_mask])
    num_finished_batch = np.sum(train_timeline.finished[col_mask])
    num_canceled_batch = np.sum(train_timeline.finished[col_mask] == False)
    return ideal_thpt, real_thpt, avg_batch_time, avg_batch_size, num_finished_batch, num_canceled_batch

def read_dataframe(log_dir: Path, train_timeline_name: str, infer_timeline_name: str, require: bool = False) -> Optional[tuple[pd.DataFrame, pd.DataFrame]]:
    train_timeline = log_dir / f"{train_timeline_name}"
    infer_timeline = log_dir / f"{infer_timeline_name}"
    if train_timeline.exists() and infer_timeline.exists():
        infer_timeline = pd.read_csv(infer_timeline)
        train_timeline = pd.read_csv(train_timeline)
        batch_name = r'batch_(\d+)_(\d+)_(\d+)'
        train_timeline = train_timeline[train_timeline['name'].str.match(batch_name)]
        batch_info = {}
        batch_info['timestamp'] = train_timeline['timestamp']
        batch_info['epoch'] = train_timeline['name'].str.extract(batch_name)[0].astype(int)
        batch_info['batch'] = train_timeline['name'].str.extract(batch_name)[1].astype(int)
        batch_info['batch_size'] = train_timeline['name'].str.extract(batch_name)[2].astype(int)
        batch_info['duration'] = train_timeline['duration']
        batch_info['start_time'] = train_timeline['timestamp']
        batch_info['end_time'] = train_timeline['timestamp'] + train_timeline['duration']
        batch_info['tag'] = train_timeline['tag']
        batch_info['finished'] = train_timeline['tag'] == 'finish'
        return pd.DataFrame(batch_info), infer_timeline
    elif not require:
        return None 
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
    df = read_dataframe(log_dir, train_timeline_name, infer_timeline_name)
    if df is None:
        for sub_log_dir in sorted(filter(lambda file_or_dir: file_or_dir.is_dir(), log_dir.iterdir()), key=lambda must_dir: str(must_dir)):
            try:
                df = read_dataframe(sub_log_dir, train_timeline_name, infer_timeline_name, require=True)
                ideal_thpt, real_thpt, avg_batch_time, avg_batch_size, num_finished_batch, num_canceled_batch = eval_throughput(*df)
                print(f"[{sub_log_dir.name}] thpt: {real_thpt:.2f} / {ideal_thpt:.2f} it/sec, avg_batch_time {avg_batch_time:.1f}, avg_batch_size {avg_batch_size:.1f}, num_finished {num_finished_batch} num_cancel {num_canceled_batch}\n")
            except Exception as e:
                print(f"[{sub_log_dir.name}] {e}\n")
    else:    
        ideal_thpt, real_thpt, avg_batch_time, avg_batch_size, num_finished_batch, num_canceled_batch = eval_throughput(*df)
        print(f"[{log_dir.name}] thpt: {real_thpt:.2f} / {ideal_thpt:.2f} it/sec, avg_batch_time {avg_batch_time:.1f}, avg_batch_size {avg_batch_size:.1f}, num_finished {num_finished_batch} num_cancel {num_canceled_batch}\n")
    
if __name__ == "__main__":
    main()