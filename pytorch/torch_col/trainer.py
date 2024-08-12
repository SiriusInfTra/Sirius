import torch
from torch.utils.data import DataLoader
import torch.distributed as torch_dist

import torch_col
from torch_col import ColocateAdjustL1Exception, SwitchL1Exception, \
    EngineColocateAdjustL1Exception
import torch_col.xsched

from .dataset import DynamicBatchDataset
from .util import EventManager, Event


class Trainer:
    def __init__(self, 
                 model: torch.nn.Module,
                 dynamic_dataset:DynamicBatchDataset, 
                 iter_train_fn:callable):
        self.rank = torch_col.get_train_rank()
        self.model = model
        self.dynamic_dataset = dynamic_dataset
        self.data_loader = DataLoader(
            dynamic_dataset, batch_size=None, 
            shuffle=False, pin_memory=True, drop_last=False, num_workers=0
        )
        self.iter_train_fn = iter_train_fn

    def train_one_epoch(self, epoch_idx: int):
        self._reset_epoch_stat()
        for batch_idx, batch in enumerate(self.data_loader):
            try:
                self.iter_train_fn(batch)
            except (
                ColocateAdjustL1Exception, 
                SwitchL1Exception, 
                EngineColocateAdjustL1Exception
            ) as e:
                if epoch_idx == 0 and batch_idx == 0:
                    raise RuntimeError("first micro batch could not be interrupted.")
                torch_col.info(f'epoch {epoch_idx} batch {batch_idx} dropped due to {e}')

                if isinstance(e, EngineColocateAdjustL1Exception):
                    # ad-hoc code here, handle batch is not killed on adjustment.
                    # should be move to c++ pytorch hook?
                    torch_col.xsched.kill_batch()

                if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                    # DDP Training
                    self.model.reducer.finalize_dropped_batch()
                else:
                    pass
                self.dynamic_dataset.cancel_micro_batch()

                batch_size = self.dynamic_dataset.get_batch_size(batch)
                batch_exception_event = \
                    f'batch_exception_{epoch_idx:02d}_{batch_idx:03d}_{batch_size:02d}'
                with EventManager.record_duration_event(batch_exception_event):
                    # cuda has alreadly synced
                    self.dynamic_dataset.hook.release_and_reply()
                    print(f'[{e}] batch_size: {batch_size} -> {self.dynamic_dataset.batch_size}.')

                torch_col.wait_barrier()
                nccl_backend = torch_dist.GroupMember.WORLD._get_backend(torch.device('cuda'))
                nccl_backend._restart_nccl_comm([torch.device(f'cuda:{self.rank}')])
            else:
                # batch passed, go to next batch
                self.dynamic_dataset.next_batch()
                if epoch_idx == 0 and batch_idx == 0:
                    self._default_first_batch_hook()

    def _reset_epoch_stat(self):
        pass

    def _default_first_batch_hook(self):
        if torch_col.is_enable_shared_tensor():
            torch_col.tag_model_end()

        if torch_col.is_enable_xsched():
            if torch_col.xsched.is_guess_nccl_begined():
                torch_col.xsched.guess_nccl_end()
                nccl_streams = torch.cuda.current_stream()
                assert len(nccl_streams) <= 1, f"nccl streams {nccl_streams}"
                if len(nccl_streams) == 1:
                    torch_col.xsched.register_stream(nccl_streams[0], False)
            torch_col.xsched.initial_kill_batch(0, 0)

                
                