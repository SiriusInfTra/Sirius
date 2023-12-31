import time
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, IterableDataset, get_worker_info
from typing import Iterator

import torch_col
from torch_col.hook import HookABC
from torch_col.util import TrainMode, EventManager

class CustomeDynamicBatchDataset(IterableDataset):
    def __init__(self, size, input_shape, num_class, max_batch_size, hook: HookABC) -> None:
        super().__init__()
        self.size = size
        self.input_shape = input_shape
        self.num_class = num_class
        self.max_batch_size = max_batch_size
        self.last_batch_size = None
        self.iter_idx = 0
        self.hook = hook
        self.all_inputs = torch.randn(size, *input_shape).pin_memory()
        self.all_targets = torch.randint(0, num_class, size=(size,), dtype=torch.long).pin_memory()
        print(f'Create CustomeDynamicBatchDataset, hook={type(hook)}.')

    @property
    def batch_size(self):
        if self.hook.train_mode.is_colocate():
            return self.hook.target_batch_size
        else:
            return self.max_batch_size

    def next_batch(self):
        if self.hook.train_mode == TrainMode.COLOCATE_L2:
            if self.hook._stub.cmd == torch_col.Event.kColocateAdjustL2:
                with EventManager.record_duration_event('adjust_l2'):
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

    def __iter__(self) -> Iterator:
        worker_info = get_worker_info()
        if worker_info is None or worker_info.num_workers == 1:
            while True:
                if self.iter_idx == self.size:
                    self.iter_idx = 0
                    break
                batch_size = min(self.batch_size, self.size - self.iter_idx)
                if self.hook.train_mode.is_colocate():
                    assert self.hook is not None
                    while batch_size <= 0:
                        self.hook.report_batch_size(batch_size)
                        time.sleep(1e-3)
                        if self.hook._stub.cmd == torch_col.Event.kColocateAdjustL1 or self.hook._stub.cmd == torch_col.Event.kColocateAdjustL2:
                            self.hook.reply_adjust()
                        batch_size = min(self.batch_size, self.size - self.iter_idx)
                elif self.hook.train_mode == TrainMode.TASKSWITCH_L1:
                    assert self.hook is not None
                    while self.hook._stub.cmd == torch_col.Event.kInterruptTrain:
                        time.sleep(1e-3)
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
                yield (inputs, targets)
        else:
            raise Exception("not support multi-process")
