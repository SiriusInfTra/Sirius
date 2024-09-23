from enum import Enum, IntEnum
import contextlib
from inspect import currentframe, getframeinfo
from dataclasses import dataclass
import time
from typing import Optional
import pandas as pd
import inspect
from io import StringIO

import torch
import torch_col
from ._C import TrainMode


def info(*args):
    if len(args) == 1:
        msg = args[0]
    else:
        msg = ' '.join([f'{arg}' for arg in args])
    if type(msg) != str:
        msg = f'{msg}'

    c_msg = msg.encode('utf-8')
    torch_col._C.CallGLOG_INFO(c_msg, 'torch_col', 0)


def info_with_frame(*args):
    if len(args) == 1:
        msg = args[0]
    else:
        msg = ' '.join([f'{arg}' for arg in args])
    if type(msg) != str:
        msg = f'{msg}'

    caller = inspect.getframeinfo(inspect.stack()[1][0])
    c_msg = msg.encode('utf-8')
    c_file = caller.filename.encode('utf-8')
    torch_col._C.CallGLOG_INFO(c_msg, c_file, caller.lineno)
    

def dinfo(*args):
    if len(args) == 1:
        msg = args[0]
    else:
        msg = ' '.join([f'{arg}' for arg in args])
    if type(msg) != str:
        msg = f'{msg}'

    caller = inspect.getframeinfo(inspect.stack()[1][0])
    c_msg = msg.encode('utf-8')
    c_file = caller.filename.encode('utf-8')
    torch_col._C.CallGLOG_DINFO(c_msg, c_file, caller.lineno)


def cleanup_previous_shm():
    import subprocess
    subprocess.run(["rm -f /dev/shm/colserve_shm_*"], 
                   shell=True, capture_output=False)
    subprocess.run(["rm -f /dev/shm/sem.colserve_shm_*"],
                   shell=True, capture_output=False)


class MemoryPool:
    @classmethod
    def get_memory_usage(cls):
        if torch_col.is_enable_shared_tensor():
            nbytes = torch_col.cuda_memory_pool_train_all_usage(torch_col.get_train_rank())
            return nbytes / 1024 / 1024 / 1024
        else:
            free, total = torch.cuda.mem_get_info(torch_col.get_train_rank())
            return (total - free) / 1024 / 1024 / 1024
    
    @classmethod
    def get_allocated_memory(cls):
        if torch_col.is_enable_shared_tensor():
            nbytes = torch_col.cuda_memory_pool_train_usage(torch_col.get_train_rank())
            return nbytes / 1024 / 1024 / 1024
        else:
            free, total = torch.cuda.mem_get_info(torch_col.get_train_rank())
            return (total - free) / 1024 / 1024 / 1024
    
    @classmethod
    def empty_cache(cls):
        if torch_col.is_enable_shared_tensor():
            torch_col.cuda_memory_pool_free_train_local(torch_col.get_train_rank())
        else:
            torch.cuda.empty_cache()


@dataclass
class Event:
    timestamp: int
    name: str
    loc: str
    duration: int
    tag: str
    
    def __str__(self) -> str:
        return f'{self.timestamp} {self.loc} {self.name} {self.duration} {self.tag}'
        
    
# Note: EventManager use us as time unit to avoid time collision, but duration is ms
class EventManager:
    _event_list: list[Event] = []
    _default_event_log_path = 'train-profile.csv'

    @classmethod
    def set_log_path(cls, path: str):
        cls._event_log_path = path

    @classmethod
    def record_event(cls, name: str, 
                     prev_event: Optional[Event]=None, 
                     record_loc:bool=False) -> Event:
        timestamp = torch_col.get_unix_timestamp_us()
        if record_loc:
            caller = inspect.getframeinfo(inspect.stack()[1][0])
            loc = f'{caller.filename}:{caller.lineno}'
        else:
            loc = 'none'
        if prev_event is not None:
            prev_event.duration = (timestamp - prev_event.timestamp) / 1000
        if len(name) > 0:
            event = Event(timestamp, name, loc, 0, '')
            cls._event_list.append(event)
            return event
    
    @classmethod
    @contextlib.contextmanager
    def record_duration_event(cls, name: str):
        timestamp = torch_col.get_unix_timestamp_us()
        loc = 'none'
        event = Event(timestamp, name, loc, 0, '')
        cls._event_list.append(event)
        yield
        event.duration = (torch_col.get_unix_timestamp_us() - timestamp) / 1000
        
    @classmethod
    def dump(cls, path: Optional[str]=None, train_mode: Optional[TrainMode]=None):
        if path is None:
            if len(torch_col.get_train_profile_log_path()):
                path = torch_col.get_train_profile_log_path()
            else:
                path = cls._default_event_log_path

        if train_mode is not None and train_mode.is_colocate():
            adjust_request_time_stamp = torch_col.get_adjust_request_time_stamp()
            adjust_done_time_stamp = torch_col.get_adjust_done_time_stamp()
            for ts in adjust_request_time_stamp:
                cls._event_list.append(Event(ts, 'recv_adjust', 'none', 0, ''))
            for ts in adjust_done_time_stamp:
                cls._event_list.append(Event(ts, 'adjust_done', 'none', 0, ''))
        df = pd.DataFrame(cls._event_list)
        df['timestamp'] = df['timestamp'] / 1000
        df['rank'] = torch_col.get_train_rank()

        if torch_col.get_train_rank() == 0:
            df_str = df.to_csv(index=None, float_format='%.2f') + '\n'
            for i in range(1, torch_col.get_train_world_size()):
                train_event_str = torch_col.dist.recv_msg(i)
                df_str += train_event_str + '\n'
            with open(path, 'w') as f:
                f.write(df_str)
            torch_col.info(f'[EventMananger] dump train event to {path}')
        else:
            df_str = df.to_csv(index=None, header=False, float_format='%.2f')
            torch_col.dist.send_msg(0, df_str)
        torch_col.dist.wait_barrier()
    
# event_manager = EventManager()

def initialize_sgd_optimizer(model, optimizer):
    # prepare grad, and optimizer state
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.zeros_like(p)
    for groups in optimizer.param_groups:
        for p in groups['params']:
            state = optimizer.state[p]
            state['momentum_buffer'] = torch.zeros_like(p)


# def monitor_sm_partition():
#     if torch_col._C.

# def print_sgd_optimizer(optimizer):
#     nbytes = 0
#     ptrs = ""
#     for group in optimizer.param_groups:
#         for p in group['params']:
#             assert p.is_cuda
#             ptrs += f"p::{p.data_ptr():x},"
#             nbytes += p.numel() * p.element_size()
#             if p.grad is not None:
#                 assert p.grad.is_cuda
#                 ptrs += f"g::{p.grad.data_ptr():x},"
#                 nbytes += p.grad.numel() * p.grad.element_size()
#             state = optimizer.state[p]
#             if 'momentum_buffer' in state:
#                 momentum_buffer = state['momentum_buffer']
#                 assert momentum_buffer.is_cuda
#                 ptrs += f"m::{momentum_buffer.data_ptr():x},"
#                 nbytes += momentum_buffer.numel() * momentum_buffer.element_size()
#     ptrs += f", total_bytes::{nbytes/1024/1024:.2f}mb"
#     print(ptrs, flush=True, file=sys.stderr)
#     print(f'memory usage {torch_col.MemoryPool.get_allocated_memory():.2f}mb, {torch_col.MemoryPool.get_memory_usage():.2f}mb', flush=True, file=sys.stderr)
#     # torch_col.MemoryPool.empty_cache()
#     # print(f'memory usage after empty cache {torch_col.MemoryPool.get_allocated_memory():.2f}mb, {torch_col.MemoryPool.get_memory_usage():.2f}mb', flush=True, file=sys.stderr)
