from __future__ import annotations

import torch
from torch.utils.data import DataLoader
import torch.distributed as torch_dist

import torch_col
from .colocate_ctrl import (
    ColocateAdjustL1Exception, 
    SwitchL1Exception, 
    EngineColocateAdjustL1Exception
)
import torch_col.xsched

# from .dataset import DynamicBatchDataset
from .dynamic_batch import DynamicBatchDataset, Batch
from .util import EventManager, Event
from dataclasses import dataclass

from typing import Callable, Any, Optional


@dataclass
class TrainStat:
    killed_batch: int = 0
    finished_batch: int = 0
    tried_batch: int = 0


@dataclass
class EpochStat:
    killed_batch: int = 0
    finished_batch: int = 0
    tried_batch: int = 0
    finished_sample: int = 0
    num_rollback_batch: int = 0

    killed_time: float = 0.0
    finished_time: float = 0.0


class Trainer:
    _trainer: Optional[Trainer] = None

    def __init__(self, 
                 model: torch.nn.Module,
                 iter_train_fn:Callable[[Any], torch.Tensor]):
        self.rank = torch_col.get_train_rank()
        self.model = model
        self.dynamic_dataset = torch_col.get_dynamic_dataset()
        self.batch_manager = torch_col.get_micro_batch_manager()
        self.data_loader = DataLoader(
            self.dynamic_dataset, batch_size=None, 
            shuffle=False, pin_memory=False, 
            drop_last=False, num_workers=0
        )
        self.iter_train_fn = iter_train_fn

        # stat
        self.overall_stat = TrainStat(0, 0, 0)
        self.epoch_stats = []
        self.epoch_events = []
        self._cur_epoch_stat = None

    def train_one_epoch(self, epoch_idx: int) -> Optional[float]:
        epoch_event_name = f'epoch_{epoch_idx:02d}_{self.dynamic_dataset.ds_size}'
        epoch_event = EventManager.record_event(epoch_event_name)

        # epoch_stat = EpochStat()
        self._cur_epoch_stat = EpochStat()
        running_loss = 0

        for batch_idx, batch in enumerate(self.data_loader):
            try:
                self._cur_epoch_stat.tried_batch += 1
                self.overall_stat.tried_batch += 1

                # self.dynamic_dataset.record_batch_event(
                #     epoch_idx, batch_idx, 
                #     self.dynamic_dataset.get_batch_size(batch), 
                #     self.dynamic_dataset.global_batch_size
                # )
                self.batch_manager.begin_batch(
                    epoch_idx, batch_idx, batch, batch['range']
                )
                _running_loss = self.iter_train_fn(batch)
            except (
                ColocateAdjustL1Exception, 
                SwitchL1Exception, 
                EngineColocateAdjustL1Exception
            ) as exception:
                self.handle_abort_batch(epoch_idx, batch_idx, batch, exception)
            else:
                # batch passed, go to next batch
                # self.dynamic_dataset.next_batch()
                self.batch_manager.finish_batch(epoch_idx, batch_idx, batch)
                
                if epoch_idx == 0 and batch_idx == 0:
                    self._default_first_batch_callback()

                running_loss += _running_loss

                self._cur_epoch_stat.finished_batch += 1
                self._cur_epoch_stat.finished_sample += \
                    self.dynamic_dataset.get_batch_size(batch)
                self.overall_stat.finished_batch += 1

        EventManager.record_event('', epoch_event)
        self.epoch_stats.append(self._cur_epoch_stat)
        self.epoch_events.append(epoch_event)
        self._cur_epoch_stat = None

        return running_loss / self.dynamic_dataset.ds_size

    def get_overall_stat(self) -> TrainStat:
        return self.overall_stat

    def get_last_epoch_stat(self) -> Optional[EpochStat]:
        if len(self.epoch_stats) == 0:
            return None
        return self.epoch_stats[-1]
    
    def get_last_epoch_event(self) -> Optional[Event]:
        if len(self.epoch_events) == 0:
            return None
        return self.epoch_events[-1]
    
    def get_last_epoch_duration(self) -> Optional[float]:
        last_epoch_event = self.get_last_epoch_event()
        if last_epoch_event is None:
            return None
        return last_epoch_event.duration
    
    def handle_abort_batch(self, epoch_idx: int, batch_idx: int, 
                           batch: Batch, exception: Exception):
        if epoch_idx == 0 and batch_idx == 0:
            raise RuntimeError("first micro batch could not be interrupted.")
        
        # --------------------------
        # first, finish abort batch
        # Ref: [Note: fast training memory adjust]

        # torch_col.info(f'epoch {epoch_idx} batch {batch_idx} dropped due to {exception}')
        torch_col.dist.wait_barrier()

        if isinstance(exception, EngineColocateAdjustL1Exception):
            # ad-hoc code here, handle batch is not killed on adjustment.
            # should be move to c++ pytorch hook?
            torch_col.xsched.kill_batch()

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            # DDP Training
            self.model.reducer.finalize_dropped_batch()
        else:
            pass
        self.batch_manager.abort_batch(epoch_idx, batch_idx, batch)

        # ---------------------------------------------
        # second, release memory and reply to inference

        batch_size = self.dynamic_dataset.get_batch_size(batch)
        batch_exception_event = \
            f'batch_exception_{epoch_idx:02d}_{batch_idx:03d}_{batch_size:02d}'
        with EventManager.record_duration_event(batch_exception_event):
            # cuda has alreadly synced
            self.dynamic_dataset.col_ctrl.release_and_reply()
            print(f'[{exception}] batch_size: {batch_size} -> '
                  f'{self.dynamic_dataset.col_ctrl.unpub_target_batch_size}.')

        self._cur_epoch_stat.killed_batch += 1
        self.overall_stat.killed_batch += 1
        
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.model.require_forward_param_sync = True

        # --------------------------------------------------
        # third, re-configure training and re-start training

        if torch_col.get_train_rank() == 0:
            torch_col.dist._DynamicBatchDistirbutor.distribute_batch(True, True)
            self.dynamic_dataset.col_ctrl.set_killed_batch_reconfiged()
        torch_col.dist.wait_barrier()

        nccl_backend = torch_dist.GroupMember.WORLD._get_backend(torch.device('cuda'))
        # restart nccl will let all training cpu sync
        nccl_backend._restart_nccl_comm([torch.device(f'cuda:{self.rank}')])

        if isinstance(self.dynamic_dataset.col_ctrl, torch_col.ColocateCtrl):
            # Ref: [Note: kill batch]
            self.dynamic_dataset.col_ctrl.set_killed_batch_recover()

        torch_col.dist.wait_barrier()

    def _default_first_batch_callback(self):
        if torch_col.is_enable_shared_tensor():
            torch_col.tag_model_end()

        if torch_col.is_enable_xsched():
            if torch_col.xsched.is_guess_nccl_begined():
                torch_col.xsched.guess_nccl_end()
                nccl_streams = torch_col.xsched.get_nccl_streams()
                assert len(nccl_streams) <= 1, f"nccl streams {nccl_streams}"
                if len(nccl_streams) == 1:
                    torch_col.xsched.register_stream(nccl_streams[0], False)
            torch.cuda.current_stream().synchronize()
            torch_col.xsched.initial_kill_batch(0, 0)

