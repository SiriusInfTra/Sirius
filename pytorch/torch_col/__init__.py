import torch_col.xsched
import torch_col.xsched
from ._C import *
from ._C import _dist as dist

from .util import (
    MemoryPool, TrainMode, EventManager, 
    info, dinfo
)
from .accumulate import GradAccumulator

# from .dataset import DynamicBatchDataset
from .colocate_ctrl import (
    register_saved_tensor_hook, create_colocate_ctrl, 
    ColocateCtrlHookMode, DummyCtrl, SwitchCtrl, 
    SwitchL1Exception, ColocateCtrl, ColocateAdjustL1Exception, 
    EngineColocateAdjustL1Exception
)
from .trainer import Trainer
from .dyanmic_batch import (
    DynamicBatchDataset, MicroBatchManager,
    get_dynamic_dataset, get_micro_batch_manager,
    init_dynamic_batch
)

from .debug_server import DebugServer


def setup_colocate_training(rank: int, world_size: int,
                            will_use_nccl_comm: bool, 
                            init_nccl_process_group: bool):
    import torch
    import torch_col
    
    if init_nccl_process_group and not will_use_nccl_comm:
        raise ValueError('init_nccl_process_group is True but will_use_nccl_comm is False')

    torch_col.torch_col_init(rank, world_size)
    if torch_col.is_enable_xsched() and will_use_nccl_comm:
        torch_col.xsched.guess_nccl_begin()

    torch.cuda.set_device(rank)
    stream = torch.cuda.Stream(device=rank)

    if torch_col.is_enable_xsched():
        torch_col.xsched.register_stream(stream)
        torch_col.info(f'xsched register stream {stream.cuda_stream}')
    else:
        torch_col.info(f'skip xsched register stream {stream.cuda_stream}')
    torch.cuda.set_stream(stream)

    if init_nccl_process_group:
        import torch.distributed as torch_dist
        if torch_dist.GroupMember.WORLD is not None:
            raise ValueError('torch.distributed already initialized')
        torch_dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch_dist.GroupMember.WORLD._get_backend(torch.device('cuda'))._set_as_default_pg()
        torch_col.dinfo(f'init_nccl_process_group done')


def cleanup_colocate_training(destory_nccl_process_group:bool):
    import torch_col

    if torch_col.is_enable_xsched():
        torch_col.xsched.unregister_all_streams()

    if destory_nccl_process_group:
        import torch.distributed as torch_dist
        torch_dist.destroy_process_group()

# torch_col_init()