from __future__ import annotations
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np


def eval_throughput(train_timeline: pd.DataFrame, infer_timeline: pd.DataFrame):
    infer_begin = np.min(infer_timeline.start_time)
    infer_end = np.max(infer_timeline.end_time) 
    col_mask = np.logical_and(train_timeline.start_time >= infer_begin, train_timeline.end_time <= infer_end)
    col_finished_mask = np.logical_and(col_mask, train_timeline.finished == True)
    num_image = np.sum(train_timeline.batch_size[col_finished_mask])
    elapsed_time = np.max(train_timeline.end_time[col_mask]) - np.min(train_timeline.start_time[col_mask])
    real_thpt = num_image / elapsed_time * 1000
    ideal_thpt = np.sum(train_timeline.batch_size[col_finished_mask]) / np.sum(train_timeline.end_time[col_finished_mask] - train_timeline.start_time[col_finished_mask]) * 1000
    return ideal_thpt, real_thpt

def read_dataframe(log_dir: Path, train_timeline_name: str, infer_timeline_name: str, require: bool = False) -> Optional[tuple[pd.DataFrame, pd.DataFrame]]:
    train_timeline = log_dir / f"{train_timeline_name}.csv"
    infer_timeline = log_dir / f"{infer_timeline_name}"
    if train_timeline.exists() and infer_timeline.exists():
        return pd.read_csv(train_timeline), pd.read_csv(infer_timeline)
    elif not require:
        return None 
    else:
        raise RuntimeError(f"{train_timeline} or {infer_timeline} does not exist.")
    

def main():
    parser = ArgumentParser()
    parser.add_argument("--log-dir", type=Path, help="log dir", required=True)
    parser.add_argument("--train-profile", type=str, help="train timeline csv file path", default="train-timeline")
    parser.add_argument("--infer-timeline", type=str, help="train timeline csv file path", default="infer-timeline")
    args = parser.parse_args()
    log_dir: Path = args.log_dir
    train_timeline_name = args.train_timeline
    infer_timeline_name = args.infer_timeline
    df = read_dataframe(log_dir, train_timeline_name, infer_timeline_name)
    if df is None:
        for sub_log_dir in sorted(filter(lambda file_or_dir: file_or_dir.is_dir(), log_dir.iterdir()), key=lambda must_dir: str(must_dir)):
            df = read_dataframe(sub_log_dir, train_timeline_name, infer_timeline_name, require=True)
            ideal_thpt, real_thpt = eval_throughput(*df)
            print(f"{sub_log_dir.name} thpt: {real_thpt:.2f}/{ideal_thpt:.2f} it/sec")
    else:    
        ideal_thpt, real_thpt = eval_throughput(*df)
        print(f"{log_dir.name} thpt: {real_thpt:.2f}/{ideal_thpt:.2f} it/sec")
    
if __name__ == "__main__":
    main()