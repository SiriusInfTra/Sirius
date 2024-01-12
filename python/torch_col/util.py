from enum import Enum, IntEnum
import contextlib
from inspect import currentframe, getframeinfo
from dataclasses import dataclass
from typing import Optional
import pandas as pd


import torch
import torch_col


class TrainMode(Enum):
    NORMAL = 'normal'
    COLOCATE_L1 = 'colocate-l1'
    COLOCATE_L2 = 'colocate-l2'
    TASKSWITCH_L0 = 'taskswitch-l0'
    TASKSWITCH_L1 = 'taskswitch-l1'
    TASKSWITCH_L2 = 'taskswitch-l2'
    TASKSWITCH_L3 = 'taskswitch-l3'
    
    def is_normal(self):
        return self == TrainMode.NORMAL
    
    def is_colocate(self):
        return self in {TrainMode.COLOCATE_L1, TrainMode.COLOCATE_L2}

    def is_kill_batch(self):
        return self in {TrainMode.COLOCATE_L1, TrainMode.TASKSWITCH_L1}

    def is_taskswitch(self):
        return self in {TrainMode.TASKSWITCH_L1, TrainMode.TASKSWITCH_L2, TrainMode.TASKSWITCH_L3}


class MemoryPool:
    @classmethod
    def get_memory_usage(cls):
        if torch_col.use_shared_tensor():
            return torch_col.cuda_memory_pool_train_all_usage() / 1024 / 1024 / 1024
        else:
            free, total = torch.cuda.mem_get_info(0)
            return (total - free) / 1024 / 1024 / 1024
    
    @classmethod
    def empty_cache(cls):
        if torch_col.use_shared_tensor():
            torch_col.cuda_memory_pool_free_train_local()
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
    event_list: list[Event] = []
    event_log_path = 'event_record.csv'

    @classmethod
    def set_log_path(cls, path: str):
        cls.event_log_path = path
    
    @classmethod
    def record_event(cls, name: str, prev_event: Optional[Event]=None) -> Event:
        timestamp = torch_col.get_unix_timestamp_us()
        # frameinfo = inspect.getframeinfo(inspect.currentframe().f_back)
        # loc = f'{frameinfo.filename}:{frameinfo.lineno}'
        loc = 'none'
        if prev_event is not None:
            prev_event.duration = (timestamp - prev_event.timestamp) / 1000
        if len(name) > 0:
            event = Event(timestamp, name, loc, 0, '')
            cls.event_list.append(event)
            return event
    
    @classmethod
    @contextlib.contextmanager
    def record_duration_event(cls, name: str):
        timestamp = torch_col.get_unix_timestamp_us()
        loc = 'none'
        event = Event(timestamp, name, loc, 0, '')
        cls.event_list.append(event)
        yield
        event.duration = (torch_col.get_unix_timestamp_us() - timestamp) / 1000
        
    @classmethod
    def dump(cls, path: Optional[str]=None, train_mode: Optional[TrainMode]=None):
        if path is None:
            path = cls.event_log_path
        if train_mode is not None and train_mode.is_colocate():
            adjust_request_time_stamp = torch_col.get_adjust_request_time_stamp()
            adjust_done_time_stamp = torch_col.get_adjust_done_time_stamp()
            for ts in adjust_request_time_stamp:
                cls.event_list.append(Event(ts, 'recv_adjust', 'none', 0, ''))
            for ts in adjust_done_time_stamp:
                cls.event_list.append(Event(ts, 'adjust_done', 'none', 0, ''))
        df = pd.DataFrame(cls.event_list)
        df['timestamp'] = df['timestamp'] / 1000
        df.to_csv(path, index=None)

    
# event_manager = EventManager()