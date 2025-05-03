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
import time


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

        self._last_max_batch_size_from_last_kill = None


    def train_one_epoch(self, epoch_idx: int) -> Optional[float]:
        epoch_event_name = f'epoch_{epoch_idx:02d}_{self.dynamic_dataset.ds_size}'
        epoch_event = EventManager.record_event(epoch_event_name)

        # epoch_stat = EpochStat()
        self._cur_epoch_stat = EpochStat()
        running_loss = 0

        for batch_idx, batch in enumerate(self.data_loader):
            self._cur_epoch_stat.tried_batch += 1
            self.overall_stat.tried_batch += 1

            if self._last_max_batch_size_from_last_kill is None:
                self._last_max_batch_size_from_last_kill = len(batch)
            elif len(batch) > self._last_max_batch_size_from_last_kill:
                torch_col.MemoryPool.empty_cache()    
                print(f'empty cache due to greater batch size {self._last_max_batch_size_from_last_kill} -> {len(batch)}')
                self._last_max_batch_size_from_last_kill = len(batch)
            
            self.batch_manager.begin_batch(
                    epoch_idx, batch_idx, batch, batch['range']
                )

            # if batch.should_update_param():
            #     torch_col.info(f'epoch {epoch_idx} batch {batch_idx} w/ sync')

            try:
                _running_loss = self.iter_train_fn(batch)
            except (
                ColocateAdjustL1Exception, 
                SwitchL1Exception, 
                EngineColocateAdjustL1Exception
            ) as exception:
                if self._is_ddp_training() and batch.should_update_param():
                    self.batch_manager.vote_abort_last_micro_batch()
                self.handle_abort_batch(epoch_idx, batch_idx, batch, exception)
            else:
                # batch passed, go to next batch
                # self.dynamic_dataset.next_batch()

                # consensus on the sync batch. 
                # should not enter this block? as we have already synced before optimizer.step(). however, task switch may execute this block
                if self._is_ddp_training() and batch.should_update_param():
                    if not self.batch_manager.vote_finish_last_micro_batch():
                        self.handle_abort_batch(
                            epoch_idx, batch_idx, batch, 
                            EngineColocateAdjustL1Exception("vote finish failed")
                        )
                        continue

                self.batch_manager.finish_batch(epoch_idx, batch_idx, batch)
                
                if epoch_idx == 0 and batch_idx == 0:
                    self._default_first_batch_callback()

                running_loss += _running_loss

                self._cur_epoch_stat.finished_batch += 1
                self._cur_epoch_stat.finished_sample += \
                    self.dynamic_dataset.get_batch_size(batch)
                self.overall_stat.finished_batch += 1

                # if batch.should_update_param():
                #     torch_col.info(f'epoch {epoch_idx} batch {batch_idx} w/ sync done')

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
            raise RuntimeError("first micro batch could not be interrupted, please increase the delay time")
        
        # -------------------------
        # first, finish abort batch
        # Ref: [Note: fast training memory adjust]

        # torch_col.info(f'epoch {epoch_idx} batch {batch_idx} dropped due to {exception}')
        torch_col.dist.wait_barrier()

        if isinstance(exception, EngineColocateAdjustL1Exception):
            # ad-hoc code here, handle batch is not killed on adjustment.
            # should be move to c++ pytorch hook?
            torch_col.xsched.kill_batch()

        if self._is_ddp_training(): # DDP Training
            self.model.reducer.finalize_dropped_batch()
        else:
            pass
        self.batch_manager.abort_batch(epoch_idx, batch_idx, batch)

        if isinstance(self.dynamic_dataset.col_ctrl, torch_col.SwitchCtrl):
            if torch_col.is_train_master():
                self.dynamic_dataset.col_ctrl._stub.set_global_has_batch_killed(True)
            torch_col.dist.wait_barrier()

        # ---------------------------------------------
        # second, release memory and reply to inference

        batch_size = self.dynamic_dataset.get_batch_size(batch)
        batch_exception_event = \
            f'batch_exception_{epoch_idx:02d}_{batch_idx:03d}_{batch_size:02d}'
        with EventManager.record_duration_event(batch_exception_event):
            # cuda has alreadly synced
            self.dynamic_dataset.col_ctrl.release_and_reply(True) # barrier to ensure consistent status
            print(f'[Rank {torch_col.get_train_rank()} | {exception}] batch_size: {batch_size} -> '
                  f'{self.dynamic_dataset.col_ctrl.unpub_target_batch_size}.')

        self._cur_epoch_stat.killed_batch += 1
        self.overall_stat.killed_batch += 1
        
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.model.require_forward_param_sync = True

        # --------------------------------------------------
        # third, re-configure training and re-start training

        train_restart_event = f'batch_restart_{epoch_idx:02d}_{batch_idx:03d}'
        with EventManager.record_duration_event(train_restart_event):
            if torch_col.get_train_rank() == 0:
                # batch across worker contain at least one sync batch 
                # or has no sync batch, it is ok to reset all vote counter
                self.batch_manager.reset_last_micro_batch_finish_vote()
            torch_col.dist.wait_barrier()

            if self._is_ddp_training():
                nccl_backend = torch_dist.GroupMember.WORLD._get_backend(torch.device('cuda'))
                # restart nccl will let all training cpu sync
                nccl_backend._restart_nccl_comm([torch.device(f'cuda:{self.rank}')])

            if torch_col.get_colocate_train_mode().is_colocate():
                # Ref: [Note: kill batch] (colcoate)
                # torch_col.info('set_killed_batch_recover')
                self.dynamic_dataset.col_ctrl.set_killed_batch_recover()

            if torch_col.get_train_rank() == 0:
                # Colocate:
                #   because we may directly reply memory adjust 
                #   before recovering the killed batch (not killing batch),
                #   distributing batch should be after recovering
                #   to configure training for those requests to avoid infer OOM
                # Task switch:
                #   call distribute batch to set training is ongoing
                torch_col.dist._DynamicBatchDistirbutor.distribute_batch(
                    True, True, False)
                if torch_col.get_colocate_train_mode().is_colocate():
                    self.dynamic_dataset.col_ctrl.set_killed_batch_reconfiged()
            torch_col.dist.wait_barrier()
            # torch_col.info(f'epoch {epoch_idx} batch {batch_idx} handle abort end.')

        # -------------------------------------
        # [task switch] fourth, wait for resume
        if torch_col.get_colocate_train_mode().is_taskswitch():
            while not self.dynamic_dataset.col_ctrl._stub.prepare_resume():
                time.sleep(1e-3)
            torch_col.dist.wait_barrier()
            # torch_col.info(f'epoch {epoch_idx} batch {batch_idx} resume training.')

        self._last_max_batch_size_from_last_kill = None

    def _default_first_batch_callback(self):
        if torch_col.is_enable_shared_tensor():
            torch_col.tag_model_end()

        if torch_col.is_enable_xsched():
            if torch_col.xsched.is_guess_nccl_begined():
                torch_col.xsched.guess_nccl_end()
                nccl_streams = torch_col.xsched.get_nccl_streams()
                assert len(nccl_streams) <= 1, f"nccl streams {nccl_streams}"
                if len(nccl_streams) == 1:
                    # torch_col.info(f'nccl stream {nccl_streams[0]}')
                    torch_col.xsched.register_stream(nccl_streams[0], False)
            torch.cuda.current_stream().synchronize()
            torch_col.xsched.initial_kill_batch(0, 0)

    def _default_first_epoch_callback(self):
        pass

    def _is_ddp_training(self):
        return isinstance(self.model, torch.nn.parallel.DistributedDataParallel)

