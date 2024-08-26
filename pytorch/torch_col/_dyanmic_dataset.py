from __future__ import annotations

from enum import IntEnum
import sys
import time
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from typing import Iterator, Optional, List, Tuple
import dataclasses

import torch_col
from torch_col.hook import HookABC, HookMode
from torch_col.util import TrainMode, MemoryPool, EventManager
import torch_col.xsched


_BATCH_FINISH_TAG = 'finish'
_BATCH_CANCEL_TAG = 'cancel'


class DatasetType(IntEnum):
    VISION = 0
    TEXT_GEN = 1
    TEXT_CLS = 2
    DYNAMIC_BATCH = 3


class DatasetState(IntEnum):
    INIT = 0
    ITER = 1
    NEXT = 2


@dataclasses.dataclass
class VisionDatasetConfig:
    input_shape: tuple
    num_class: int

@dataclasses.dataclass
class TextDatasetConfig:
    seq_len: int    


class Batch(dict):
    def __init__(self):
        super().__init__()

    def will_do_step(self):
        try:
            return self['do_step']
        except KeyError:
            return True
        
    def __len__(self) -> int:
        return DynamicBatchDataset._dynamic_batch_dataset.get_batch_size(self)
        
        

class MicroBatchManager:
    '''
    MicroBatchManager is used to manage the micro batch,
    such as finishing or canceling the micro batch.
    '''
    _micro_batch_manager = None

    def __init__(self, 
                 dynamic_dataset: DynamicBatchDataset,
                 enable_grad_accumulate: bool = False,
                 checkpoint_micro_batch: bool = False):
        assert MicroBatchManager._micro_batch_manager is None

        MicroBatchManager._micro_batch_manager = self 
        self._dynamic_dataset = dynamic_dataset
        self._enable_grad_accumulate = enable_grad_accumulate
        self._checkpoint_micro_batch = checkpoint_micro_batch

        self._last_batch = None
        self._micro_batch_indices_in_global_batch = []

    def scale_loss(self, batch, loss: torch.Tensor):
        if self._enable_grad_accumulate:
            batch_size = self._dynamic_dataset.get_batch_size(batch)
            scale_factor = self._dynamic_dataset.global_batch_size / batch_size
            loss /= scale_factor
            return loss
        else:
            return loss
        
    def optimizer_step(
        self, 
        batch: Batch,
        optimizer:torch.optim.Optimizer, 
        amp_scaler: Optional[torch.cuda.amp.GradScaler] = None,
        grad_accumulator: Optional[torch_col.accumulate.GradAccumulator] = None
    ):
        def _step():
            if amp_scaler is not None:
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                optimizer.step()

        if self.is_do_checkpoint_micro_batch():
            assert grad_accumulator is not None
            with self._dynamic_dataset.hook.steps_no_interrupt():
                step_event = EventManager.record_event('optimizer_step')
                grad_accumulator.accumulate()
                if batch.will_do_step():
                    grad_accumulator.step(optimizer, amp_scaler)
                    step_event.tag = 'global'
                else:
                    step_event.tag = 'local'
                EventManager.record_event('', step_event)
        else:
            if batch.will_do_step():
                with self._dynamic_dataset.hook.steps_no_interrupt():
                    step_event = EventManager.record_event('optimizer_step')
                    _step()
                    EventManager.record_event('', step_event)

    def begin_micro_batch(self, 
                          batch: Batch, 
                          batch_index: List[Tuple[int, int]]):
        self._micro_batch_indices_in_global_batch.append(batch_index)
        self._last_batch = batch

    def finish_micro_batch(self, batch: Batch):
        pass

    def abort_micro_batch(self, batch: Batch):
        pass

    def _next_global_batch(self):
        pass


class DynamicBatchDataset(IterableDataset):
    '''
    DynamicBatchDataset only manage the data and 
    return the batch with dynamic batch size to training.
    '''
    _dynamic_batch_dataset = None

    def __init__(self, 
                 ds_type: DatasetType, 
                 ds_size: int,
                 global_batch_size: int,
                 hook: HookABC,
                 vision_dataset_config: Optional[VisionDatasetConfig] = None,
                 text_dataset_config: Optional[TextDatasetConfig] = None,
                 enable_grad_accumulate: bool = False,
                 checkpoint_micro_batch: bool = False,
                 empty_cache_at_larger_batch_size: bool = False,
                 fake_data=False):
        assert DynamicBatchDataset._dynamic_batch_dataset is None
        DynamicBatchDataset._dynamic_batch_dataset = self

        super().__init__()
        self.ds_type = ds_type
        self.ds_size = ds_size
        self.global_batch_size = global_batch_size
        self.hook = hook
        self.vision_dataset_config = vision_dataset_config
        self.text_dataset_config = text_dataset_config

        self._last_batch = None
        self._enable_grad_accumulate = enable_grad_accumulate
        self._checkpoint_micro_batch = checkpoint_micro_batch
        self._empty_cache_at_larger_batch_size = empty_cache_at_larger_batch_size
        self._state = DatasetState.INIT
        self._global_batch_event = None
        self._batch_event = None

        torch_col.dist.init_dynamic_batch(ds_size, global_batch_size)

        if not ds_type == DatasetType.VISION:
            fake_data = True
            print('Warning: fake_data is forced to be True for non-vision dataset', 
                  file=sys.stderr)

        if not fake_data:
            if ds_type == DatasetType.VISION:
                self.all_inputs = {
                    'images': torch.from_numpy(
                            np.load('workload_data/cifiar10/cifiar10_inputs.npy')
                        ).pin_memory(),
                    'labels': torch.from_numpy(
                            np.load('workload_data/cifiar10/cifiar10_targets.npy')
                        ).pin_memory()
                }
                assert (self.vision_dataset_config.num_class 
                        == torch.max(self.all_inputs['labels']).item() + 1), \
                    f"expect num of class: {torch.max(self.all_inputs['labels']).item() + 1}."
                assert ds_size == len(self.all_inputs['images']), \
                    f"expect size {len(self.all_inputs['image'])}."
                assert (self.vision_dataset_config.input_shape 
                        == self.all_inputs['images'].shape[1:]), \
                    f"expect input shape: {self.all_inputs['images'].shape[1:]}"
            else:
                raise Exception(f"Unsupported dataset type: {ds_type}")
        else:
            if ds_type == DatasetType.VISION:
                self.all_inputs = {
                    'images': torch.randn(
                            ds_size, *vision_dataset_config.input_shape
                        ).pin_memory(),
                    'labels': torch.randint(
                            0, vision_dataset_config.num_class, (ds_size,)
                        ).pin_memory()
                }
            elif ds_type == DatasetType.TEXT_GEN:
                self.all_inputs = {
                    "input_ids": torch.from_numpy(
                            np.random.randint(100, 30000, 
                                              (ds_size, self.text_dataset_config.seq_len))
                        ).pin_memory(),
                }
                self.all_inputs['labels'] = self.all_inputs['input_ids']

    def get_target_batch_size(self):
        if self.hook.train_mode.is_colocate():
            return self.hook.target_batch_size
        else:
            self.hook.batch_size

    def get_next_batch_size(self):
        return torch_col.dist._DynamicBatchDistirbutor.query_next_batch_size()
                
    def get_batch_size(self, batch: Batch):
        if self.ds_type == DatasetType.VISION:
            return len(batch['images'])
        elif self.ds_type == DatasetType.TEXT_GEN:
            return len(batch['input_ids'])
        else:
            raise Exception(f"Unsupported dataset type: {self.ds_type}")

    def record_batch_event(self, epoch_idx:int, batch_idx:int, 
                           batch_size:int, global_batch_size=None):
        self._batch_event = EventManager.record_event(
            f'batch_{epoch_idx:02d}_{batch_idx:03d}_{batch_size:02d}'
        )
        if self._enable_grad_accumulate:
            if self.global_batch_event is None:
                assert global_batch_size is not None
                self._global_batch_event = EventManager.record_event(
                    f'global_batch_{epoch_idx:02d}_{batch_idx:03d}_{global_batch_size:02d}'
                )

    def _retrieve_batch(self, batch_index: List[Tuple[int, int]]) -> dict:
        batch = {}
        for k in self.all_inputs.keys():
            input_list = []
            for i, j in batch_index:
                input_list.append(self.all_inputs[k][i:j])
            batch[k] = torch.cat(input_list, dim=0).pin_memory()
        return batch

    def _wait_valid_batch_size(self):
        batch_size = self.get_next_batch_size()
        if self.hook.train_mode.is_colocate():
            while batch_size <= 0:
                # self.hook.report_batch_size(batch_size)
                torch_col.update_current_batch_size(batch_size)
                if (self.hook._stub.cmd == torch_col.CtrlEvent.kColocateAdjustL1
                    or self.hook._stub.cmd == torch_col.CtrlEvent.kColocateAdjustL2):
                    self.hook.release_and_reply()
                time.sleep(1e-3)
                batch_size = self.get_next_batch_size()
        elif self.hook.train_mode == TrainMode.TASKSWITCH_L1:
            while self.hook._stub.cmd == torch_col.CtrlEvent.kInterruptTrain:
                self.hook.switch()
                time.sleep(1e-3)
        return batch_size    

    def _get_batch(self) -> Batch:
        _batch_size = self._wait_valid_batch_size()
        batch_index, last_micro_batch = \
                torch_col.dist._DynamicBatchDistirbutor.get_batch(self.batch_size)
        batch = self._retrieve_batch(batch_index)
        batch_size = self.get_batch_size(batch)

        assert batch_size == _batch_size, (
            f"batch size mismatch: {batch_size} != {_batch_size}, "
            f"check batch distirbutor and dataset implementation.")

        if last_micro_batch:
            batch['do_step'] = False
        else:
            batch['do_step'] = True
        return batch

    def __iter__(self) -> Iterator[Batch]:
        get_last_micro_batch = False

        while not get_last_micro_batch:
            batch = self._get_batch()
            batch_size = self.get_batch_size(batch)

            if batch.will_do_step():
                get_last_micro_batch = True
  
            if (self._state != DatasetState.INIT
                and torch_col.is_enable_shared_tensor()
                and self._empty_cache_at_larger_batch_size 
                and self.get_batch_size(self._last_batch) < batch_size
            ):
                MemoryPool.empty_cache()

            self._state = DatasetState.ITER
            self._last_batch = batch
            if self.hook is not None:
                self.hook.report_batch_size(batch_size)
                torch_col.update_current_batch_size(batch_size)

            yield batch
        
    