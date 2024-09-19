from __future__ import annotations

from enum import IntEnum
import sys
import time
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from typing import Iterator, Optional, List, Tuple, Union
import dataclasses

import torch_col
from torch_col.colocate_ctrl import (
    CtrlBase, DummyCtrl, SwitchCtrl, 
    ColocateCtrl, ColocateCtrlHookMode,
    get_colocate_ctrl
)
from torch_col.util import TrainMode, MemoryPool, EventManager
import torch_col.xsched
from ._C._dist import _DynamicBatchDistirbutor


_BATCH_FINISH_TAG = 'finish'
_BATCH_CANCEL_TAG = 'cancel'


def _num_sample_of_batch_range_vec(batch_range_vec: List[Tuple[int, int]]) -> int:
    return sum([j - i for i, j in batch_range_vec])


def _num_sample_of_batch_range(batch_range: Tuple[int, int]) -> int:
    return batch_range[1] - batch_range[0]


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
    def __init__(self, inputs: Optional[dict] = None):
        super().__init__()
        for k, v in inputs.items():
            self[k] = v

    def should_update_param(self):
        try:
            return self['do_step']
        except KeyError:
            return True

    def get_range(self) -> List[Tuple[int, int]]:
        return self['range']
        
    def __len__(self) -> int:
        return DynamicBatchDataset._dynamic_batch_dataset.get_batch_size(self)
        

class MicroBatchManager:
    '''
    MicroBatchManager is used to manage the micro batch,
    such as finishing or canceling the micro batch.
    '''

    _micro_batch_manager: Optional[MicroBatchManager] = None

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
        self._batch_event = None
        self._global_batch_event = None
        self._past_micro_batch_range_vecs_in_global_batch = []
        self._last_batch_range_vec = None


    def scale_loss(self, batch: Batch, loss: torch.Tensor):
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
        optimizer: torch.optim.Optimizer, 
        amp_scaler: Optional[torch.cuda.amp.GradScaler] = None,
        grad_accumulator: Optional[torch_col.accumulate.GradAccumulator] = None
    ):
        if self._should_checkpoint_micro_batch():
            assert grad_accumulator is not None
            with get_colocate_ctrl().steps_no_interrupt():
                step_event = EventManager.record_event('optimizer_step')
                grad_accumulator.accumulate()
                if batch.should_update_param():
                    grad_accumulator.step(optimizer, amp_scaler)
                    step_event.tag = 'global'
                else:
                    step_event.tag = 'local'
                EventManager.record_event('', step_event)
        else:
            def _step():
                if amp_scaler is not None:
                    amp_scaler.step(optimizer)
                    amp_scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            if batch.should_update_param():
                with get_colocate_ctrl().steps_no_interrupt():
                    step_event = EventManager.record_event('optimizer_step')
                    _step()
                    EventManager.record_event('', step_event)

    def begin_batch(self, 
                    epoch_idx: int,
                    batch_idx: int,
                    batch: Batch, 
                    batch_range_vec: List[Tuple[int, int]]):
        self._last_batch_range_vec = batch_range_vec
        self._last_batch = batch
        self._record_batch_event(epoch_idx, batch_idx, len(batch))
        
    def finish_batch(self, epoch_idx: int, batch_idx: int, batch: Batch):
        if batch.should_update_param():
            torch.cuda.current_stream().synchronize()

        col_ctrl = get_colocate_ctrl()
        if col_ctrl.train_mode == TrainMode.COLOCATE_L2:
            if col_ctrl._stub.cmd == torch_col.CtrlEvent.kColocateAdjustL2:
                col_ctrl.release_and_reply()
        
        self._complete_batch_event(_BATCH_FINISH_TAG)
        self._past_micro_batch_range_vecs_in_global_batch.append(
            self._last_batch_range_vec)

        # if self._enable_grad_accumulate:
        #     if self._should_checkpoint_micro_batch():
        #         _DynamicBatchDistirbutor.finish_batch()
        # else:
        #     _DynamicBatchDistirbutor.finish_batch()

        if (not self._enable_grad_accumulate 
            or self._should_checkpoint_micro_batch()
        ):
            _DynamicBatchDistirbutor.finish_batch(
                batch.get_range(), batch.should_update_param())

        # micro batch update param implys the end of global batch
        if batch.should_update_param():
            self._next_global_batch()
        
        self._last_batch = None
        self._last_batch_range_vec = None
        self._dynamic_dataset._next_batch()

    def abort_batch(self, epoch_idx: int, batch_idx: int, batch: Batch):
        self._complete_batch_event(_BATCH_CANCEL_TAG)
        if self._enable_grad_accumulate and not self._checkpoint_micro_batch:
            self._rollback_micro_batch()
        else:
            _DynamicBatchDistirbutor.abort_batch(self._last_batch_range_vec)
        self._last_batch = None
        self._last_batch_range_vec = None

    def _record_batch_event(self, epoch_idx: int, batch_idx: int, 
                            batch_size: int):
        self._batch_event = EventManager.record_event(
            f'batch_{epoch_idx:02d}_{batch_idx:03d}_{batch_size:02d}'
        )

        if self._enable_grad_accumulate:
            if self._global_batch_event is None:
                global_batch_size = _DynamicBatchDistirbutor.get_global_batch_size()
                assert global_batch_size is not None
                self._global_batch_event = EventManager.record_event(
                    f'global_batch_{epoch_idx:02d}_{batch_idx:03d}_{global_batch_size:02d}'
                )

    def _complete_batch_event(self, tag: str):
        assert self._batch_event is not None
        self._batch_event.tag = tag
        EventManager.record_event('', self._batch_event)
        self._batch_event = None

    def _complate_global_batch_event(self, tag: str):
        assert self._global_batch_event is not None
        self._global_batch_event.tag = tag
        EventManager.record_event('', self._global_batch_event)
        self._global_batch_event = None

    def _should_checkpoint_micro_batch(self):
        if self._enable_grad_accumulate and self._checkpoint_micro_batch:
            return True
        else:
            return False
        
    def _rollback_micro_batch(self):
        assert self._enable_grad_accumulate and not self._checkpoint_micro_batch, (
            "only used for accumulation without checkpoint")
        
        self._complate_global_batch_event(_BATCH_CANCEL_TAG)
        for idx in self._past_micro_batch_range_vecs_in_global_batch:
            torch_col.Trainer._trainer._cur_epoch_stat.num_rollback_batch += \
                _num_sample_of_batch_range(idx)
            _DynamicBatchDistirbutor.abort_batch(idx)

        _DynamicBatchDistirbutor.abort_batch(self._last_batch_range_vec)
        torch_col.Trainer._trainer._cur_epoch_stat.num_rollback_batch += \
            _num_sample_of_batch_range(self._last_batch_range_vec)
        
        self._past_micro_batch_range_vecs_in_global_batch = []
        self._last_batch_range_vec = None

    def _next_global_batch(self):
        if self._enable_grad_accumulate:
            # when grad accumulate is not enabled, 
            # global batch event is the same as batch event,
            # we don't need to record global batch event.
            self._complate_global_batch_event(_BATCH_FINISH_TAG)

        if (self._enable_grad_accumulate 
            and not self._checkpoint_micro_batch
        ):
            # for idx in self._past_micro_batch_range_vecs_in_global_batch:
            num_micro_batch = len(self._past_micro_batch_range_vecs_in_global_batch)
            for i in range(num_micro_batch):
                batch_range = self._past_micro_batch_range_vecs_in_global_batch[i]
                end_of_global_batch = i == num_micro_batch - 1
                _DynamicBatchDistirbutor.finish_batch(batch_range, end_of_global_batch)

        self._past_micro_batch_range_vecs_in_global_batch = []
        _DynamicBatchDistirbutor.next_global_batch()


class DynamicBatchDataset(IterableDataset):
    '''
    DynamicBatchDataset only manage the data and 
    return the batch with dynamic batch size to training.
    '''

    _dynamic_batch_dataset: Optional[DynamicBatchDataset] = None

    def __init__(self, 
                 ds_type: DatasetType, 
                 ds_size: int,
                #  global_batch_size: int,
                 colocate_ctrl: Union[DummyCtrl, SwitchCtrl, ColocateCtrl],
                 vision_dataset_config: Optional[VisionDatasetConfig] = None,
                 text_dataset_config: Optional[TextDatasetConfig] = None,
                #  enable_grad_accumulate: bool = False,
                #  checkpoint_micro_batch: bool = False,
                 empty_cache_at_larger_batch_size: bool = False,
                 fake_data=False):
        assert DynamicBatchDataset._dynamic_batch_dataset is None
        DynamicBatchDataset._dynamic_batch_dataset = self

        super().__init__()
        self.ds_type = ds_type
        self.ds_size = ds_size
        # self.global_batch_size = global_batch_size
        self.col_ctrl = colocate_ctrl
        self.vision_dataset_config = vision_dataset_config
        self.text_dataset_config = text_dataset_config

        self._empty_cache_at_larger_batch_size = empty_cache_at_larger_batch_size
        self._state = DatasetState.INIT

        # self._last_batch = None
        # self._enable_grad_accumulate = enable_grad_accumulate
        # self._checkpoint_micro_batch = checkpoint_micro_batch
        
        # self._global_batch_event = None
        # self._batch_event = None

        # torch_col.dist.init_dynamic_batch_distributor(ds_size, global_batch_size)

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
        # will be input_batch_size or 
        # dynamic batch size determined by inf-tra control
        return self.col_ctrl.target_batch_size
    
        # if self.col_ctrl.train_mode.is_colocate():
        #     return self.col_ctrl.target_batch_size
        # else:
        #     self.col_ctrl.input_batch_size

    def get_next_batch_size(self):
        return torch_col.dist._DynamicBatchDistirbutor.query_next_batch_size(
                self.get_target_batch_size())
                
    def get_batch_size(self, batch: Batch):
        if self.ds_type == DatasetType.VISION:
            return len(batch['images'])
        elif self.ds_type == DatasetType.TEXT_GEN:
            return len(batch['input_ids'])
        else:
            raise Exception(f"Unsupported dataset type: {self.ds_type}")

    def _next_batch(self):
        assert self._state == DatasetState.ITER
        self._state = DatasetState.NEXT

    def _retrieve_batch(self, batch_range_vec: List[Tuple[int, int]]) -> dict:
        batch = {}

        batch_size = _num_sample_of_batch_range_vec(batch_range_vec)
        for k in self.all_inputs.keys():
            batch[k] = torch.empty([batch_size, *self.all_inputs[k].shape[1:]], 
                                   dtype=self.all_inputs[k].dtype,
                                   device=f'cuda:{torch_col.get_train_rank()}')
            off = 0
            for i, j in batch_range_vec:
                batch[k][off:off+(j-i)] = self.all_inputs[k][i:j]
                off += j - i

        return batch

    def _wait_valid_batch_size(self):
        batch_size = self.get_next_batch_size()
        if self.col_ctrl.train_mode.is_colocate():
            while batch_size <= 0:
                # self.hook.report_batch_size(batch_size)
                torch_col.update_current_batch_size(batch_size)
                if (self.col_ctrl._stub.cmd == torch_col.CtrlEvent.kColocateAdjustL1
                    or self.col_ctrl._stub.cmd == torch_col.CtrlEvent.kColocateAdjustL2
                ):
                    self.col_ctrl.release_and_reply()
                time.sleep(1e-3)
                batch_size = self.get_next_batch_size()
        elif self.col_ctrl.train_mode == TrainMode.TASKSWITCH_L1:
            while self.col_ctrl._stub.cmd == torch_col.CtrlEvent.kInterruptTrain:
                self.col_ctrl.switch()
                time.sleep(1e-3)
        return batch_size    

    def _get_batch(self) -> Batch:
        _batch_size = self._wait_valid_batch_size()
        batch_range_vec, last_micro_batch = \
            torch_col.dist._DynamicBatchDistirbutor.get_batch(_batch_size)
        _batch = self._retrieve_batch(batch_range_vec)

        # update batch size, as there maybe not enough samples
        batch = Batch(_batch)
        batch_size = self.get_batch_size(batch)

        assert batch_size == _batch_size, (
            f"batch size mismatch: {batch_size} != {_batch_size}, "
            f"check batch distirbutor and dataset implementation.")

        if last_micro_batch:
            batch['do_step'] = True
        else:
            batch['do_step'] = False
        batch['range'] = batch_range_vec

        return batch

    def __iter__(self) -> Iterator[Batch]:
        # num_gotten_global_batch = 0
        num_global_batch_per_epoch = \
            _DynamicBatchDistirbutor.get_num_global_batch_per_epoch()
        
        epoch_idx = _DynamicBatchDistirbutor.next_epoch()

        _DynamicBatchDistirbutor.next_global_batch()

        # torch_col.info(f'[dynamic dataset] epoch {epoch_idx} start')

        while (_DynamicBatchDistirbutor.get_num_proced_global_batch() 
               < num_global_batch_per_epoch
        ):
            batch = self._get_batch()
            batch_size = self.get_batch_size(batch)

            # if batch.should_update_param():
            #     num_gotten_global_batch += 1
  
            if (self._state != DatasetState.INIT
                and torch_col.is_enable_shared_tensor()
                and self._empty_cache_at_larger_batch_size 
                and self.get_batch_size(self._last_batch) < batch_size
            ):
                MemoryPool.empty_cache()

            self._state = DatasetState.ITER
            self._last_batch = batch
            if self.col_ctrl is not None:
                self.col_ctrl.report_batch_size(batch_size)
                torch_col.update_current_batch_size(batch_size)

            yield batch


def init_dynamic_batch(
    dataset_size: int,
    dataset_type: DatasetType,
    dataset_config: Union[VisionDatasetConfig, TextDatasetConfig],
    batch_size: int, 
    global_batch_size: Optional[int] = None,
    checkpoint_micro_batch: bool = False,
    empty_cache_at_larger_batch_size: bool = False,
    fake_data: bool = False
) -> Tuple[DynamicBatchDataset, MicroBatchManager]:
    assert torch_col.is_configured(), "torch_col is not initialized"
    assert torch_col.colocate_ctrl.CtrlBase._colocate_ctrl is not None, (
        "colocate control must be initialized before dynamic batch dataset."
    )

    enable_grad_accumulate = (
        global_batch_size is not None
        and global_batch_size > batch_size * torch_col.get_train_world_size()
    )
    
    if not enable_grad_accumulate and checkpoint_micro_batch:
        print(f'torch_col: Warning: should not accumulate grad '
              f'(global_batch_size {global_batch_size}, '
              f'batch_size {batch_size}), '
              f'checkpoint_micro_batch will be False.',
              file=sys.stderr)
        checkpoint_micro_batch = False

    if global_batch_size is None:
        global_batch_size = batch_size * torch_col.get_train_world_size()

    vision_ds_config = None
    text_ds_config = None
    if dataset_type == DatasetType.VISION:
        vision_ds_config = dataset_config
    elif dataset_type == DatasetType.TEXT_GEN:
        text_ds_config = dataset_config

    dynamic_dataset = DynamicBatchDataset(
        ds_type=dataset_type,
        ds_size=dataset_size,
        colocate_ctrl=torch_col.colocate_ctrl.CtrlBase._colocate_ctrl,
        vision_dataset_config=vision_ds_config,
        text_dataset_config=text_ds_config,
        empty_cache_at_larger_batch_size=empty_cache_at_larger_batch_size,
        fake_data=fake_data,
    )
    torch_col.dist.init_dynamic_batch_distributor(
        dataset_size, batch_size, global_batch_size)

    mirco_batch_manager = MicroBatchManager(
        dynamic_dataset=dynamic_dataset,
        enable_grad_accumulate=enable_grad_accumulate,
        checkpoint_micro_batch=checkpoint_micro_batch,
    )
    return dynamic_dataset, mirco_batch_manager


def get_dynamic_dataset() -> DynamicBatchDataset:
    assert DynamicBatchDataset._dynamic_batch_dataset is not None
    return DynamicBatchDataset._dynamic_batch_dataset


def get_micro_batch_manager() -> MicroBatchManager:
    assert MicroBatchManager._micro_batch_manager is not None
    return MicroBatchManager._micro_batch_manager

        
    