from contextlib import nullcontext
from enum import IntEnum
import os
import time
import numpy as np
import torch
import torch.distributed
from torch.utils.data import IterableDataset, get_worker_info
from typing import Iterator, Optional
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

import sys
import torch_col
from torch_col.colocate_ctrl import CtrlBase, ColocateCtrlHookMode
from torch_col.util import TrainMode, MemoryPool, EventManager
import torch_col.xsched


_BATCH_FINISH_TAG = 'finish'
_BATCH_CANCEL_TAG = 'cancel'


def _vision_task(model_name):
    return 'resnet' in model_name or 'vit' in model_name or 'swin' in model_name

def _text_gen(model_name):
    return 'gpt' in model_name

def _text_cls(model_name):
    return 'bert' in model_name


class DatasetState(IntEnum):
    INIT = 0
    ITER = 1
    NEXT = 2


class DynamicBatchDataset(IterableDataset):
    def __init__(self, model_name, size, max_batch_size,
                 hook: CtrlBase, trace: Optional[list[dict]], 
                 input_shape = None, num_class = None,
                 seq_len = None, 
                 max_global_batch_size = None,
                 empty_cache_at_larger_batch_size = False,
                 checkpoint_micro_batch = False,
                 fake_data=False) -> None:
        super().__init__()
        self.size = size
        self.model_name = model_name
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
        if not _vision_task(self.model_name):
            fake_data = True
        if not fake_data:
            if os.environ.get('COL_IMAGENET', '0') == '1':
                rng = np.random.default_rng(42)
                image = np.load('imagenet-100/imagenet-100-images.npy', mmap_mode='c')
                size = image.shape[0]
                indices = rng.permutation(size)
                n = size // torch_col.get_train_world_size()
                rank = torch_col.get_train_rank()
                image = image[indices[n*rank:n*(rank+1)]]
                label = np.load('imagenet-100/imagenet-100-labels.npy', mmap_mode='c')
                label = label[indices[n*rank:n*(rank+1)]]
                self.all_inputs = {
                    # 'image': torch.from_numpy(np.load('workload_data/cifiar10/cifiar10_inputs.npy')).pin_memory(),
                    # 'label': torch.from_numpy(np.load('workload_data/cifiar10/cifiar10_targets.npy')).pin_memory(),
                    'image': torch.from_numpy(image).pin_memory(),
                    'label': torch.from_numpy(label).pin_memory(),
                } 
                self.size = len(self.all_inputs['label'])           
                self.num_class = 100
            elif os.environ.get('COL_CIFAR100', '0') == '1':
                # rng = np.random.default_rng(42)
                rng = np.random.default_rng()
                transform = transforms.Compose([
                    transforms.ToTensor(),
                ])

                # 仅下载训练数据集
                train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
                size = len(train_dataset)
                # 打乱数据索引
                indices = rng.permutation(size)

                # 计算每个进程需要处理的数据量
                n = size // torch_col.get_train_world_size()
                rank = torch_col.get_train_rank()

                # 根据当前进程的 rank 获取该进程需要处理的数据索引
                subset_indices = indices[n*rank:n*(rank+1)]
                subset_dataset = Subset(train_dataset, subset_indices)

                # 创建 DataLoader
                train_loader = DataLoader(subset_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

                # 提取图像和标签数据
                all_images = []
                all_labels = []

                for images, labels in train_loader:
                    all_images.append(images)
                    all_labels.append(labels)

                # 拼接所有批次的数据
                all_images = torch.cat(all_images)
                all_labels = torch.cat(all_labels)

                # 将图像和标签数据转换为 PyTorch 张量并固定到内存中
                self.all_inputs = {
                    'image': all_images.pin_memory(),
                    'label': all_labels.pin_memory(),
                }

                # 设置数据大小和类别数量
                self.size = len(self.all_inputs['label'])
                self.num_class = 100
            elif _vision_task(self.model_name):
                self.all_inputs = {
                    'image': torch.from_numpy(np.load('workload_data/cifar10/cifar10_inputs.npy')).pin_memory(),
                    'label': torch.from_numpy(np.load('workload_data/cifar10/cifar10_targets.npy')).pin_memory()
                }
                assert num_class == torch.max(self.all_inputs['label']).item() + 1, \
                    f"expect num of class: {torch.max(self.all_inputs['label']).item() + 1},."
                assert size == len(self.all_inputs['image']), f"expect size {len(self.all_inputs['image'])}."
                assert input_shape == self.all_inputs['image'].shape[1:], \
                    f"expect input shape: {self.all_inputs['image'].shape[1:]}"
            else:
                raise Exception("not support model")
            # self.all_inputs = torch.from_numpy(np.load('workload_data/cifar10/cifar10_inputs.npy')).pin_memory()
            # self.all_targets = torch.from_numpy(np.load('workload_data/cifar10/cifar10_targets.npy')).pin_memory()
        else:
            if _vision_task(self.model_name):
                self.all_inputs = {
                    'image': torch.randn(size, *input_shape).pin_memory(),
                    'label': torch.randint(0, num_class, size=(size,), dtype=torch.long).pin_memory()
                }
                # self.all_inputs = torch.randn(size, *input_shape).pin_memory()
                # self.all_targets = torch.randint(0, num_class, size=(size,), dtype=torch.long).pin_memory()
            elif _text_gen(self.model_name):
                self.all_inputs = {
                    "input_ids": torch.from_numpy(np.random.randint(100, 30000, (size, seq_len))).pin_memory(),
                }
                self.all_inputs['labels'] = self.all_inputs['input_ids']
        self.trace = trace
        self.trace_idx = 0
        if trace is not None:
            assert hook.train_mode == TrainMode.NORMAL and hook.hook_mode == ColocateCtrlHookMode.NONE, \
                'only normal train can valid trace.'
        self.empty_cache_at_larger_batch_size = empty_cache_at_larger_batch_size
        print(f'Create DynamicBatchDataset, hook={type(hook)}.')
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
        
    def get_batch_size(self, batch):
        if _vision_task(self.model_name):
            return len(batch['image'])
        elif _text_gen(self.model_name):
            return len(batch['input_ids'])
        else:
            raise Exception("not support model")

    def record_batch_event(self, epoch, i, batch_size, global_batch_size=None):
        self.batch_event = EventManager.record_event(
            f'batch_{epoch:02d}_{i:03d}_{batch_size:02d}'
        )
        if self.enable_accumulation:
            if self.global_batch_event is None:
                assert global_batch_size is not None
                self.global_batch_event = EventManager.record_event(
                    f'global_batch_{epoch:02d}_{i:03d}_{global_batch_size:02d}'
                )

    def next_batch(self):
        if self.is_do_step():
            torch.cuda.current_stream().synchronize()

        if self.hook.train_mode == TrainMode.COLOCATE_L2:
            if self.hook._stub.cmd == torch_col.CtrlEvent.kColocateAdjustL2:
                self.hook.release_and_reply()

        assert self.state == DatasetState.ITER
        self.batch_event.tag = _BATCH_FINISH_TAG
        EventManager.record_event('', self.batch_event)
        self.batch_event = None
        if self.enable_accumulation and self.at_global_batch_end():
            self.global_batch_event.tag = _BATCH_FINISH_TAG
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
        if torch_col.is_enable_shared_tensor() and self.last_batch_size != self.batch_size:
            torch.cuda.synchronize()
            torch_col.MemoryPool.empty_cache() # to avoid memory fragmentation

    def at_global_batch_end(self):
        assert self.enable_accumulation, "only used for accumulation"
        # print(f'{self.accumulate_iter_idx}, {self.last_batch_size}, {self.global_batch_size} ')
        return self.accumulate_iter_idx + self.last_batch_size == self.global_batch_size        

    def is_do_step(self):
        if self.enable_accumulation:
            return self.at_global_batch_end()
        else:
            return True
    
    def get_context(self, model: torch.nn.parallel.DistributedDataParallel):
        if self.enable_accumulation and not self.is_do_step():
            # torch_col.info("no sync")
            return model.no_sync()
        else:
            # torch_col.info("sync")
            return nullcontext()
        
    def is_do_checkpoint_micro_batch(self):
        if self.enable_accumulation and self.checkpoint_micro_batch:
            return True
        else:
            return False

    def rollback_micro_batch(self):
        assert self.enable_accumulation and not self.checkpoint_micro_batch, \
            "only used for accumulation without checkpoint"
        assert self.micro_batch_iter_idx >= self.global_batch_iter_idx, \
            f"{self.micro_batch_iter_idx} vs {self.global_batch_iter_idx}"
        self.num_rollback_samples_in_epoch += self.micro_batch_iter_idx - self.global_batch_iter_idx
        self.micro_batch_iter_idx = self.global_batch_iter_idx
        self.accumulate_iter_idx = 0

        self.global_batch_event.tag = _BATCH_CANCEL_TAG
        EventManager.record_event('', self.global_batch_event)
        self.global_batch_event = None
    
    def cancel_micro_batch(self):
        self.batch_event.tag = _BATCH_CANCEL_TAG
        EventManager.record_event('', self.batch_event)
        # self.batch_event = None # to be check 
        if self.enable_accumulation and not self.checkpoint_micro_batch:
            self.rollback_micro_batch()

    def scale_loss(self, loss):
        if self.enable_accumulation:
            scale_factor = self.global_batch_size / self.last_batch_size
            # print('scale_factor=', scale_factor)
            loss /= scale_factor
            # loss = loss * self.last_batch_size / self.global_batch_size
        else:
            pass

    def step_stage(self, hook:torch_col.colocate_ctrl.CtrlBase, 
                   optimizer: torch.optim.Optimizer, 
                   scaler: Optional[torch.cuda.amp.GradScaler] = None, 
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
        if os.environ.get('COL_RAND_BZ', '0') == '1':
            global np_rng
            # global torch_rng_cpu
            # global torch_rng_cuda
            if 'np_rng' not in globals():
                np_rng = np.random.default_rng(42)
                # torch_rng_cpu = torch.random.get_rng_state()
                # torch_rng_cuda = torch.cuda.random.get_rng_state()
            def _get_batch_size():
                batch_size = np_rng.integers(16, 64)
                if self.max_global_batch_size is None: # not use accumulation
                    batch_size = min(batch_size, self.size - self.micro_batch_iter_idx)
                else:
                    if self.global_batch_size is None:
                        self.global_batch_size = min(self.size - self.global_batch_iter_idx, self.max_global_batch_size)
                    batch_size = min(batch_size, self.global_batch_size - self.accumulate_iter_idx)
                return batch_size
        else:
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
                torch_col.update_current_batch_size(batch_size)
                time.sleep(1e-3)
                if (self.hook._stub.cmd == torch_col.CtrlEvent.kColocateAdjustL1
                    or self.hook._stub.cmd == torch_col.CtrlEvent.kColocateAdjustL2):
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
        if self.state != DatasetState.INIT and self.last_batch_size < batch_size and torch_col.is_enable_shared_tensor():
            if self.empty_cache_at_larger_batch_size:
                MemoryPool.empty_cache()
        self.state = DatasetState.ITER
        self.last_batch_size = batch_size
        # inputs = torch.randn(batch_size, *self.input_shape)
        # targets = torch.randint(0, self.num_class, size=(batch_size,), dtype=torch.long)
        inputs = {}
        for k in self.all_inputs.keys():
            inputs[k] = self.all_inputs[k][self.micro_batch_iter_idx:self.micro_batch_iter_idx+batch_size]
        # inputs = self.all_inputs[self.micro_batch_iter_idx:self.micro_batch_iter_idx+batch_size]
        # targets = self.all_targets[self.micro_batch_iter_idx:self.micro_batch_iter_idx+batch_size]
        if self.hook is not None:
            self.hook.report_batch_size(batch_size)
            torch_col.update_current_batch_size(batch_size)
        # print(f'micro batch iter {self.micro_batch_iter_idx} acc iter {self.accumulate_iter_idx}', file=sys.stderr, flush=True)
        # return inputs, targets
        return inputs
    
    def shuffule(self):
        indices = np.random.permutation(self.size)
        for k in self.all_inputs.keys():
            self.all_inputs[k] = self.all_inputs[k][indices]
    
    def __iter__(self) -> Iterator:
        worker_info = get_worker_info()
        assert worker_info is None or worker_info.num_workers == 1, \
            "not support multi-process"
        while True:
            if self.micro_batch_iter_idx == self.size:
                self.micro_batch_iter_idx = 0
                if self.global_batch_iter_idx is not None:
                    assert self.global_batch_iter_idx == self.size, \
                        f"{self.global_batch_iter_idx} vs {self.size}"
                    self.global_batch_iter_idx = 0
                    self.num_rollback_samples_in_epoch = 0
                break
            yield self.iter_batch()
        self.shuffule()
            