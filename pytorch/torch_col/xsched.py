from contextlib import contextmanager
from typing import Optional
import torch
from torch.cuda import Stream
import ctypes
import torch_col

__pysched_dll = ctypes.CDLL('libPySched.so', mode=ctypes.RTLD_LOCAL)
__pysched_handle_dict = {}
def register_stream(stream: Stream):
    handle: ctypes.c_uint64 = __pysched_dll.RegisterStream(ctypes.c_uint64(stream.cuda_stream))
    __pysched_handle_dict[stream.cuda_stream] = handle

    torch_col._C.InitSMPartition()
    

def initial_kill_batch(epoch, batch, stream: Optional[Stream] = None):
    if epoch == 0 and batch == 0:
        t1 = torch_col.get_unix_timestamp()
        if stream is None:
            stream = torch.cuda.current_stream()
        handle = __pysched_handle_dict[stream.cuda_stream]
        num_cmds = __pysched_dll.AbortStream(handle)
        stream.synchronize()
        t2 = torch_col.get_unix_timestamp()
        print(f'initial_kill_batch cost {t2 - t1} ms, num_cmds={num_cmds}')


def kill_batch(stream: Optional[Stream] = None):
    # if torch_col.kill_batch_on_recv():
    #     return
    t1 = torch_col.get_unix_timestamp()
    if stream is None:
        stream = torch.cuda.current_stream()
    handle = __pysched_handle_dict[stream.cuda_stream]
    num_cmds = __pysched_dll.AbortStream(handle)
    stream.synchronize()
    t2 = torch_col.get_unix_timestamp()
    print(f'kill batch cost {t2 - t1} ms, num_cmds={num_cmds}')
    

def unregister_stream(stream: Optional[Stream] = None):
    if stream is None:
        stream = torch.cuda.current_stream()
    handle = __pysched_handle_dict[stream.cuda_stream]
    __pysched_dll.UnRegisterStream(handle)

# def abort_stream(stream: Optional[Stream] = None):
#     if stream is None:
#         stream = torch.cuda.current_stream()
#     __pysched_dll.AbortStream(stream.__pysched_handle)


def get_xqueue_size(stream: Optional[Stream] = None) -> int:
    if stream is None:
        stream = torch.cuda.current_stream()
    handle = __pysched_handle_dict[stream.cuda_stream]
    return int(__pysched_dll.GetXQueueSize(handle))


@contextmanager
def critical_section(stream: Optional[Stream] = None):
    if stream is None:
        stream = torch.cuda.current_stream()
    __pysched_dll.CritialSectionBegin(stream.__pysched_handle)
    yield
    __pysched_dll.CritialSectionEnd(stream.__pysched_handle)
