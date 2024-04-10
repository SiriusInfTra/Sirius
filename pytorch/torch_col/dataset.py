from enum import IntEnum
import time
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from typing import Iterator, Optional

import sys
import torch_col
from torch_col.hook import HookABC, HookMode
from torch_col.util import TrainMode, MemoryPool, EventManager

class DatasetState(IntEnum):
    INIT = 0
    ITER = 1
    NEXT = 2

class CustomeDynamicBatchDataset(IterableDataset):
    def __init__(self, size, input_shape, num_class, max_batch_size, 
                 hook: HookABC, trace: Optional[list[dict]], 
                 max_global_batch_size = None,
                 empty_cache_at_larger_batch_size = False,
                 checkpoint_micro_batch = False,
                 fake_data=False) -> None:
        super().__init__()
        self.size = size
        self.input_shape = input_shape
        self.num_class = num_class
        self.max_batch_size = max_batch_size
        self.max_global_batch_size = max_global_batch_size
        self.enable_accumulation = max_global_batch_size is not None
        self.checkpoint_micro_batch = self.enable_accumulation and checkpoint_micro_batch
        self.global_batch_size = None
        self.accumulate_iter_idx = None if max_global_batch_size is None else 0
        self.last_batch_size = None
        self.micro_batch_iter_idx = 0
        self.global_batch_iter_idx = None if max_global_batch_size is None else 0
        self.global_batch_id_in_epoch = None if max_global_batch_size is None else 0
        self.num_rollback_samples_in_epoch = None if max_global_batch_size is None else 0
        self.hook = hook
        self.state = DatasetState.INIT
        self.global_batch_event = None
        self.batch_event = None
        if not fake_data:
            self.all_inputs = torch.from_numpy(np.load('workload_data/cifiar10/cifiar10_inputs.npy')).pin_memory()
            self.all_targets = torch.from_numpy(np.load('workload_data/cifiar10/cifiar10_targets.npy')).pin_memory()
        else:
            self.all_inputs = torch.randn(size, *input_shape).pin_memory()
            self.all_targets = torch.randint(0, num_class, size=(size,), dtype=torch.long).pin_memory()
        assert num_class == torch.max(self.all_targets).item() + 1, f"expect num of class: {torch.max(self.all_targets).item() + 1}."
        assert size == len(self.all_inputs), f"expect size {len(self.all_inputs)}."
        assert input_shape == self.all_inputs.shape[1:], f"expect input shape: {self.all_inputs.shape[1:]}"
        self.trace = trace
        self.trace_idx = 0
        if trace is not None:
            assert hook.train_mode == TrainMode.NORMAL and hook.hook_mode == HookMode.NONE, 'only normal train can valid trace.'
        self.empty_cache_at_larger_batch_size = empty_cache_at_larger_batch_size
        print(f'Create CustomeDynamicBatchDataset, hook={type(hook)}.')
        print(f'enable_accumulation={self.enable_accumulation}, checkpoint_micro_batch={self.checkpoint_micro_batch}.')

    @property
    def batch_size(self):
        '''Return the batch size of the current iteration.'''
        if self.hook.train_mode.is_colocate():
            return self.hook.target_batch_size
        elif self.trace is not None:
            if self.trace_idx < len(self.trace):
                return self.trace[self.trace_idx]['batch_size']
            else:
                return 0
        else:
            return self.max_batch_size

    def record_batch_event(self, epoch, i, batch_size, global_batch_size=None):
        self.batch_event = EventManager.record_event(f'batch_{epoch:02d}_{i:03d}_{batch_size:02d}')
        if self.enable_accumulation:
            if self.global_batch_event is None:
                assert global_batch_size is not None
                self.global_batch_event = EventManager.record_event(f'global_batch_{epoch:02d}_{i:03d}_{global_batch_size:02d}')

    def next_batch(self):
        if self.is_do_step():
            torch.cuda.current_stream().synchronize()

        if self.hook.train_mode == TrainMode.COLOCATE_L2:
            if self.hook._stub.cmd == torch_col.CtrlEvent.kColocateAdjustL2:
                self.hook.release_and_reply()
                #     t0 = time.time()
                #     old_gpu_mem = gpu_mem()
                #     torch.cuda.synchronize()
                #     torch.cuda.empty_cache()
                #     hook.adjust_l2()
                #     t1 = time.time()
                #     if not use_shared_tensor:
                #         mem_info = f'gpu mem {old_gpu_mem:.1f} -> {gpu_mem():.1f}'
                #     else:
                #         mem_info = f'mem pool {torch_col.cuda_memory_pool_train_all_usage() / 1024 / 1024:.1f}M'
                #     print('batch {} adjust : bs {} -> {} | {:.1f}ms | {:.1f}ms | {}'.format(
                #         i, train_dataset.last_batch_size, train_dataset.batch_size, (time.time()-batch_begin) * 1000, (t1 - t0) * 1000, 
                #         mem_info))
        assert self.state == DatasetState.ITER
        self.batch_event.tag = 'finish'
        EventManager.record_event('', self.batch_event)
        self.batch_event = None
        if self.enable_accumulation and self.at_global_batch_end():
            self.global_batch_event.tag = 'finish'
            EventManager.record_event('', self.global_batch_event)
            self.global_batch_event = None

        self.state = DatasetState.NEXT
        self.micro_batch_iter_idx += self.last_batch_size
        if self.enable_accumulation:
            self.accumulate_iter_idx += self.last_batch_size
            if self.accumulate_iter_idx == self.global_batch_size:
                self.accumulate_iter_idx = 0
                self.global_batch_iter_idx += self.global_batch_size
                self.global_batch_size = None
        self.trace_idx += 1
        # if torch_col.use_shared_tensor():
        #     torch_col.MemoryPool.empty_cache() # to avoid memory fragmentation

    def at_global_batch_end(self):
        assert self.enable_accumulation, "only used for accumulation"
        # print(f'{self.accumulate_iter_idx}, {self.last_batch_size}, {self.global_batch_size} ')
        return self.accumulate_iter_idx + self.last_batch_size == self.global_batch_size        

    def is_do_step(self):
        if self.enable_accumulation:
            return self.at_global_batch_end()
        else:
            return True
        
    def is_do_checkpoint_micro_batch(self):
        if self.enable_accumulation and self.checkpoint_micro_batch:
            return True
        else:
            return False

    def rollback_micro_batch(self):
        assert self.enable_accumulation and not self.checkpoint_micro_batch, \
            "only used for accumulation without checkpoint"
        assert self.micro_batch_iter_idx >= self.global_batch_iter_idx, f"{self.micro_batch_iter_idx} vs {self.global_batch_iter_idx}"
        self.num_rollback_samples_in_epoch += self.micro_batch_iter_idx - self.global_batch_iter_idx
        self.micro_batch_iter_idx = self.global_batch_iter_idx
        self.accumulate_iter_idx = 0

        self.global_batch_event.tag = 'cancel'
        EventManager.record_event('', self.global_batch_event)
        self.global_batch_event = None
    
    def cancel_micro_batch(self):
        self.batch_event.tag = 'cancel'
        if self.enable_accumulation and not self.checkpoint_micro_batch:
            self.rollback_micro_batch()

    def scale_loss(self, loss):
        if self.enable_accumulation:
            scale_factor = self.global_batch_size / self.last_batch_size
            loss /= scale_factor
            # loss = loss * self.last_batch_size / self.global_batch_size
        else:
            pass

    def step_stage(self, hook:torch_col.hook.HookABC, 
                   optimizer: torch.optim.Optimizer, 
                   scaler: torch.cuda.amp.GradScaler = None, 
                   grad_accumulator:torch_col.accumulate.GradAccumulator = None):
        def _step():
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        if self.is_do_checkpoint_micro_batch():
            assert grad_accumulator is not None
            with hook.steps_no_interrupt():
                step_event = EventManager.record_event('optimizer_step')
                grad_accumulator.accumulate()
                if self.is_do_step():
                    grad_accumulator.step(optimizer, scaler)
                    step_event.tag = 'global'
                else:
                    step_event.tag = 'local'
                EventManager.record_event('', step_event)
        else:
            if self.is_do_step():
                with hook.steps_no_interrupt():
                    step_event = EventManager.record_event('optimizer_step')
                    _step()
                    EventManager.record_event('', step_event)


    def iter_batch(self) -> Iterator:
        def _get_batch_size():
            if self.max_global_batch_size is None: # not use accumulation
                batch_size = min(self.batch_size, self.size - self.micro_batch_iter_idx)
            else:
                if self.global_batch_size is None:
                    self.global_batch_size = min(self.size - self.global_batch_iter_idx, self.max_global_batch_size)
                batch_size = min(self.batch_size, self.global_batch_size - self.accumulate_iter_idx)
            return batch_size
        
        batch_size = _get_batch_size()
        if self.hook.train_mode.is_colocate():
            assert self.hook is not None
            while batch_size <= 0:
                self.hook.report_batch_size(batch_size)
                time.sleep(1e-3)
                if self.hook._stub.cmd == torch_col.CtrlEvent.kColocateAdjustL1 or self.hook._stub.cmd == torch_col.CtrlEvent.kColocateAdjustL2:
                    self.hook.release_and_reply()
                # batch_size = min(self.batch_size, self.size - self.micro_batch_iter_idx)
                batch_size = _get_batch_size()
        elif self.hook.train_mode == TrainMode.TASKSWITCH_L1:
            assert self.hook is not None
            while self.hook._stub.cmd == torch_col.CtrlEvent.kInterruptTrain:
                self.hook.switch()
                time.sleep(1e-3)
        elif self.trace is not None:
            trace_item = self.trace[self.trace_idx]
            assert batch_size == trace_item['batch_size'], f"{batch_size} vs {trace_item['batch_size']}"
        if self.state != DatasetState.INIT and self.last_batch_size < batch_size and torch_col.use_shared_tensor():
            if self.empty_cache_at_larger_batch_size:
                MemoryPool.empty_cache()
        self.state = DatasetState.ITER
        self.last_batch_size = batch_size
        # inputs = torch.randn(batch_size, *self.input_shape)
        # targets = torch.randint(0, self.num_class, size=(batch_size,), dtype=torch.long)
        inputs = self.all_inputs[self.micro_batch_iter_idx:self.micro_batch_iter_idx+batch_size]
        targets = self.all_targets[self.micro_batch_iter_idx:self.micro_batch_iter_idx+batch_size]
        if self.hook is not None:
            self.hook.report_batch_size(batch_size)
        # print(f'micro batch iter {self.micro_batch_iter_idx} acc iter {self.accumulate_iter_idx}', file=sys.stderr, flush=True)
        return inputs, targets
    
    def __iter__(self) -> Iterator:
        worker_info = get_worker_info()
        assert worker_info is None or worker_info.num_workers == 1, "not support multi-process"
        while True:
            if self.micro_batch_iter_idx == self.size:
                self.micro_batch_iter_idx = 0
                if self.global_batch_iter_idx is not None:
                    assert self.global_batch_iter_idx == self.size, f"{self.global_batch_iter_idx} vs {self.size}"
                    self.global_batch_iter_idx = 0
                    self.num_rollback_samples_in_epoch = 0
                break
            yield self.iter_batch()
            