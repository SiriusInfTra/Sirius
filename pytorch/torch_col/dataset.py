import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, IterableDataset, get_worker_info
from typing import Iterator, Optional

import torch_col
from torch_col.hook import HookABC, HookMode
from torch_col.util import TrainMode, EventManager

class CustomeDynamicBatchDataset(IterableDataset):
    def __init__(self, size, input_shape, num_class, max_batch_size, hook: HookABC, trace: Optional[list[dict]]) -> None:
        super().__init__()
        self.size = size
        self.input_shape = input_shape
        self.num_class = num_class
        self.max_batch_size = max_batch_size
        self.last_batch_size = None
        self.iter_idx = 0
        self.hook = hook
        self.all_inputs = torch.from_numpy(np.load('workload_data/cifiar10/cifiar10_inputs.npy')).pin_memory()
        self.all_targets = torch.from_numpy(np.load('workload_data/cifiar10/cifiar10_targets.npy')).pin_memory()
        assert num_class == torch.max(self.all_targets).item() + 1, f"expect num of class: {torch.max(self.all_targets).item() + 1}."
        assert size == len(self.all_inputs), f"expect size {len(self.all_inputs)}."
        assert input_shape == self.all_inputs.shape[1:], f"expect input shape: {self.all_inputs.shape[1:]}"
        self.trace = trace
        self.trace_idx = 0
        if trace is not None:
            assert hook.train_mode == TrainMode.NORMAL and hook.hook_mode == HookMode.NONE, 'only normal train can valid trace.'
        print(f'Create CustomeDynamicBatchDataset, hook={type(hook)}.')

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

    def next_batch(self):
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
        assert self.last_batch_size is not None
        self.iter_idx += self.last_batch_size
        self.last_batch_size = None
        self.trace_idx += 1
        # if torch_col.use_shared_tensor():
        #     torch_col.MemoryPool.empty_cache() # to avoid memory fragmentation
        
    def iter_batch(self) -> Iterator:
        batch_size = min(self.batch_size, self.size - self.iter_idx)
        if self.hook.train_mode.is_colocate():
            assert self.hook is not None
            while batch_size <= 0:
                self.hook.report_batch_size(batch_size)
                time.sleep(1e-3)
                if self.hook._stub.cmd == torch_col.CtrlEvent.kColocateAdjustL1 or self.hook._stub.cmd == torch_col.CtrlEvent.kColocateAdjustL2:
                    self.hook.release_and_reply()
                batch_size = min(self.batch_size, self.size - self.iter_idx)
        elif self.hook.train_mode == TrainMode.TASKSWITCH_L1:
            assert self.hook is not None
            while self.hook._stub.cmd == torch_col.CtrlEvent.kInterruptTrain:
                self.hook.switch()
                time.sleep(1e-3)
        elif self.trace is not None:
            trace_item = self.trace[self.trace_idx]
            assert batch_size == trace_item['batch_size'], f"{batch_size} vs {trace_item['batch_size']}"
        # self.iter_idx += batch_size
        if self.hook.train_mode == TrainMode.NORMAL or self.hook.train_mode == TrainMode.COLOCATE_L2:
            assert self.last_batch_size is None
        self.last_batch_size = batch_size
        # inputs = torch.randn(batch_size, *self.input_shape)
        # targets = torch.randint(0, self.num_class, size=(batch_size,), dtype=torch.long)
        inputs = self.all_inputs[self.iter_idx:self.iter_idx+batch_size]
        targets = self.all_targets[self.iter_idx:self.iter_idx+batch_size]
        if self.hook is not None:
            self.hook.report_batch_size(batch_size)
        return inputs, targets
    
    def iter_batch_with_trace(self) -> Iterator:
        trace_item = self.trace[self.trace_idx]
        batch_size = trace_item['batch_size']
        inputs = self.all_inputs[self.iter_idx:self.iter_idx+batch_size]
        targets = self.all_targets[self.iter_idx:self.iter_idx+batch_size]
        self.last_batch_size = batch_size
        return inputs, targets
    
    def __iter__(self) -> Iterator:
        worker_info = get_worker_info()
        assert worker_info is None or worker_info.num_workers == 1, "not support multi-process"
        while True:
            if self.iter_idx == self.size:
                self.iter_idx = 0
                break
            yield self.iter_batch()
            