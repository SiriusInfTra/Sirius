from contextlib import contextmanager
from typing import Optional, Union
import torch
from torch.cuda import Stream
import torch_col

# __pysched_dll = ctypes.CDLL('libPySched.so', mode=ctypes.RTLD_LOCAL)
# __pysched_handle_dict = {}
# def register_stream(stream: Stream):
#     handle: ctypes.c_uint64 = __pysched_dll.RegisterStream(ctypes.c_uint64(stream.cuda_stream))
#     __pysched_handle_dict[stream.cuda_stream] = handle

#     torch_col._C.SMPartitionInit()
    

# def initial_kill_batch(epoch, batch, stream: Optional[Stream] = None):
#     if epoch == 0 and batch == 0:
#         t1 = torch_col.get_unix_timestamp()
#         if stream is None:
#             stream = torch.cuda.current_stream()
#         handle = __pysched_handle_dict[stream.cuda_stream]
#         num_cmds = __pysched_dll.AbortStream(handle)
#         stream.synchronize()
#         t2 = torch_col.get_unix_timestamp()
#         print(f'initial_kill_batch cost {t2 - t1} ms, num_cmds={num_cmds}')


# def kill_batch(stream: Optional[Stream] = None):
#     # if torch_col.kill_batch_on_recv():
#     #     return
#     t1 = torch_col.get_unix_timestamp()
#     if stream is None:
#         stream = torch.cuda.current_stream()
#     handle = __pysched_handle_dict[stream.cuda_stream]
#     num_cmds = __pysched_dll.AbortStream(handle)
#     stream.synchronize()
#     t2 = torch_col.get_unix_timestamp()
#     print(f'kill batch cost {t2 - t1} ms, num_cmds={num_cmds}')
    

# def unregister_stream(stream: Optional[Stream] = None):
#     if stream is None:
#         stream = torch.cuda.current_stream()
#     handle = __pysched_handle_dict[stream.cuda_stream]
#     __pysched_dll.UnRegisterStream(handle)

# # def abort_stream(stream: Optional[Stream] = None):
# #     if stream is None:
# #         stream = torch.cuda.current_stream()
# #     __pysched_dll.AbortStream(stream.__pysched_handle)


# def get_xqueue_size(stream: Optional[Stream] = None) -> int:
#     if stream is None:
#         stream = torch.cuda.current_stream()
#     handle = __pysched_handle_dict[stream.cuda_stream]
#     return int(__pysched_dll.GetXQueueSize(handle))


# @contextmanager
# def critical_section(stream: Optional[Stream] = None):
#     if stream is None:
#         stream = torch.cuda.current_stream()
#     __pysched_dll.CritialSectionBegin(stream.__pysched_handle)
#     yield
#     __pysched_dll.CritialSectionEnd(stream.__pysched_handle)


def register_stream(stream: Union[Stream, int], 
                    is_enable_dynamic_sm_partition: bool = True):
    if isinstance(stream, Stream):
        stream = stream.cuda_stream
    torch_col.RegisterStream(stream)

    if (
        torch_col.has_colocated_infer_server()
        and torch_col.is_enable_dynamic_sm_partition()
        and is_enable_dynamic_sm_partition
    ):
        torch_col._C.SMPartitionInit(stream)


def unregister_stream(stream: Union[Stream, int]):
    if isinstance(stream, Stream):
        stream = stream.cuda
    torch_col.UnRegisterStream(stream)


def unregister_all_streams():
    torch_col.UnRegisterAllStreams()


def get_xqueue_size(stream: Optional[Union[Stream, int]] = None) -> int:
    if stream is None:
        stream = torch.cuda.current_stream()
    if isinstance(stream, Stream):
        stream = stream.cuda_stream
    return torch_col.GetXQueueSize(stream)


def initial_kill_batch(epoch, batch, 
                       stream: Optional[Union[Stream, int]] = None):
    if epoch == 0 and batch == 0:
        t1 = torch_col.get_unix_timestamp()
        # if stream is None:
        #     stream = torch.cuda.current_stream()
        # num_cmds = torch_col.AbortStream(stream.cuda_stream)
        if stream is None:
            num_cmds = torch_col.AbortAllStreams()
            torch_col.SyncAllStreams()
        else:
            if isinstance(stream, Stream):
                stream = stream.cuda_stream
            num_cmds = torch_col.AbortStream(stream)
            stream.synchronize()
        t2 = torch_col.get_unix_timestamp()
        print(f'initial_kill_batch cost {t2 - t1} ms, num_cmds={num_cmds}')


def kill_batch(stream: Optional[Stream] = None):
    # if torch_col.kill_batch_on_recv():
    #     return
    t1 = torch_col.get_unix_timestamp()
    if stream is None:
        num_cmds = torch_col.AbortAllStreams()
        torch_col.SyncAllStreams()
    else:
        if isinstance(stream, Stream):
            stream = stream.cuda_stream
        num_cmds = torch_col.AbortStream(stream)
        torch_col.SyncStream(stream)
    t2 = torch_col.get_unix_timestamp()
    print(f'kill batch cost {t2 - t1} ms, num_cmds={num_cmds} '
          f'total_queue_size={torch_col.GetTotalXQueueSize()}')


def set_reject_cuda_calls(reject: bool):
    torch_col.SetRejectCudaCalls(reject)


def guess_nccl_begin():
    torch_col.GuessNcclBegin()


def guess_nccl_end():
    torch_col.GuessNcclEnd()


def is_guess_nccl_begined():
    return torch_col.IsGuessNcclBegined()


def get_nccl_streams():
    return torch_col.GetNcclStreams()

