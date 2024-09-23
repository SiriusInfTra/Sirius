# cython: c_string_type=unicode, c_string_encoding=utf8
# distutils: language = c++

from enum import Enum


class ColocateCtrlHookMode(Enum):
    NONE = 'none'
    SYNC = 'sync'
    # XSCHED_ASYNC_SIGNAL = 'xsched-async-signal'  
    XSCHED_SYNC = 'xsched-sync'
    XSCHED_SYNC2 = 'xsched-sync2'

    def use_xsched(self):
        return self in {ColocateCtrlHookMode.XSCHED_SYNC, 
                        ColocateCtrlHookMode.XSCHED_SYNC2}


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
        return self in {TrainMode.TASKSWITCH_L1, TrainMode.TASKSWITCH_L2, 
                        TrainMode.TASKSWITCH_L3}