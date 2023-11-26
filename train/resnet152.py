from typing import Iterator, Optional, Sized
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import models
from torchvision.datasets import FakeData
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Sampler, IterableDataset, get_worker_info
import time
import argparse
import os, sys

import torch_col

# TODO: wrapp in torch_col package
class SwitchL1Exception(Exception):
    pass

class SwitchHook:
    def __init__(self, mode) -> None:
        self.stub = torch_col.PySwitchStub()
        self.mode = mode

    def get_hook(self):
        def hook(module, input, output):
            torch.cuda.synchronize()
            if self.stub.cmd == torch_col.Event.kInterruptTrain:
                raise SwitchL1Exception("[Task Switch]")
            elif self.stub.cmd == torch_col.Event.kResumeTrain:
                self.stub.cmd = None
        return hook
    
    def stop(self):
        self.stub.stop()


class ColocateAdjustL1Exception(Exception):
    pass

class ColocateHook:
    def __init__(self, batch_size) -> None:
        self.stub = torch_col.PyColocateStub(batch_size)

    def try_reply_adjust_l1(self):
        if self.stub.cmd == torch_col.Event.kColocateAdjustL1:
            self.stub.adjust_l1_done()

    def try_reply_adjust_l2(self):
        if self.stub.cmd == torch_col.Event.kColocateAdjustL2:
            self.stub.adjust_l2_done()

    def reply_adjust_l1(self):
        assert self.stub.cmd == torch_col.Event.kColocateAdjustL1
        self.stub.adjust_l1_done()

    def get_hook(self):
        def hook(module, input, output):
            torch.cuda.synchronize()
            # print(f'{module} {time.time()}')
            if self.stub.cmd == torch_col.Event.kColocateAdjustL1:
                # print(f'receive kColocateAdjustL1 {time.time()}')
                raise ColocateAdjustL1Exception("[Colocate Adjust L1]")
        return hook
    
    def stop(self):
        self.stub.stop()


def register_fbward_hook(module:torch.nn.Module, hook):
    if len(list(module.children())) == 0:
        module.register_forward_hook(hook)
        module.register_backward_hook(hook)
    else:
        for child in module.children():
            register_fbward_hook(child, hook)

    
def gpu_mem():
    free, total = torch.cuda.mem_get_info(0)
    return (total - free) / 1024 / 1024 / 1024


class CustomeDynamicBatchDataset(IterableDataset):
    def __init__(self, size, input_shape, num_class, max_batch_size, mode, hook) -> None:
        super().__init__()
        self.size = size
        self.input_shape = input_shape
        self.num_class = num_class
        self.max_batch_size = max_batch_size
        self.last_batch_size = None
        self.iter_idx = 0
        self.mode = mode
        self.hook = hook
        self.all_inputs = torch.randn(size, *input_shape).pin_memory()
        self.all_targets = torch.randint(0, num_class, size=(size,), dtype=torch.long).pin_memory()

    @property
    def batch_size(self):
        if self.mode == 'colocate-l1' or self.mode == 'colocate-l2':
            return self.hook.stub.target_batch_size
        else:
            return self.max_batch_size

    def next_batch(self):
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
                if self.mode == 'colocate-l1' or self.mode == 'colocate-l2':
                    assert self.hook is not None
                    while batch_size <= 0:
                        time.sleep(1e-3)
                        self.hook.try_reply_adjust_l1()
                        self.hook.try_reply_adjust_l2()
                elif self.mode == 'task-switch-l1':
                    assert self.hook is not None
                    while self.hook.stub.cmd == torch_col.Event.kInterruptTrain:
                        time.sleep(1e-3)
                # self.iter_idx += batch_size
                if self.mode == 'normal' or self.mode == 'colocate-l2':
                    assert self.last_batch_size is None
                self.last_batch_size = batch_size
                # inputs = torch.randn(batch_size, *self.input_shape)
                # targets = torch.randint(0, self.num_class, size=(batch_size,), dtype=torch.long)
                inputs = self.all_inputs[self.iter_idx:self.iter_idx+batch_size]
                targets = self.all_targets[self.iter_idx:self.iter_idx+batch_size]
                yield (inputs, targets)
        else:
            raise Exception("not support multi-process")


def train(num_epoch=10, batch_size=256, mode='normal', **kargs):
    hook = kargs["hook"]
    
    ori_batch_size = batch_size

    model = models.resnet101()
    model = model.cuda(0)

    use_shared_tensor = os.environ.get('USE_SHARED_TENSOR', '0') == '1'

    if use_shared_tensor:
        print("Train params memory usage: {:.2f}M".format(torch_col.cuda_memory_pool_train_usage() / 1024 / 1024))

    criterion = nn.CrossEntropyLoss().cuda(0)
    optimizer = torch.optim.SGD(model.parameters(), 0.1, 
                                momentum=0.9, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    if use_shared_tensor:
        print("Train after init memory pool usage: {:.2f}M".format(torch_col.cuda_memory_pool_train_usage() / 1024 / 1024))

    # dummy data, todo: learning rate auto scaling
    train_dataset = CustomeDynamicBatchDataset(1000, (3, 224, 224), 
                                               50, batch_size, mode, hook)
    train_loader = DataLoader(train_dataset, batch_size=None, 
                              shuffle=False, pin_memory=True, drop_last=False, num_workers=0)
    
    if mode == 'task-switch-l1' or mode == 'colocate-l1':
        register_fbward_hook(model, hook.get_hook())
    print(f"train in {mode} mode")

    total_killed_batch = 0
    total_finished_batch = 0
    total_tried_batch = 0

    model.train()
    for epoch in range(num_epoch):
        begin = time.time()
        batch_cnt = 0
        killed_batch = 0
        finished_batch = 0
        tried_batch = 0
        killed_time = 0
        finished_time = 0
        wait_bs_valid_sec = 0 # add infer may cause batch size <= 0
        for i, (images, targets) in enumerate(train_loader):
            # print(f"Batch {i}, batch size {len(images)} {train_dataset.batch_size}")
            images:torch.Tensor = images.to('cuda:0', non_blocking=True)
            targets:torch.Tensor = targets.to('cuda:0', non_blocking=True)
            
            # micro_batch_size = random.randint(1, batch_size)
            # print(micro_batch_size)
            batch_begin = time.time()
            micro_batch_begin = None
            try:
                # print('hook.cmd', hook.stub.cmd, flush=True)
                micro_batch_begin = time.time()
                tried_batch += 1
                total_tried_batch += 1
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
                # finished_time += time.time() - micro_batch_begin
                finished_batch += 1
                total_finished_batch += 1
                batch_cnt += 1
            except (ColocateAdjustL1Exception, SwitchL1Exception) as e:
                killed_batch += 1
                total_killed_batch += 1
                # killed_time += time.time() - micro_batch_begin
                t0 = time.time()
                torch.cuda.synchronize()
                old_gpu_mem = gpu_mem()
                torch.cuda.empty_cache()
                t1 = time.time()
                if isinstance(e, ColocateAdjustL1Exception):
                    hook.stub.adjust_l1_done()
                if not use_shared_tensor:
                    mem_info = f'gpu mem {old_gpu_mem:.1f} -> {gpu_mem():.1f}'
                else:
                    mem_info = f'mem pool {torch_col.cuda_memory_pool_train_usage() / 1024 / 1024:.1f}M'
                print('{} free cache {:.1f}ms, new micro batch size {}, {}'.format(
                    e, (t1 - t0) * 1000, train_dataset.batch_size, mem_info))
            else:
                if mode == 'colocate-l2' and hook.stub.cmd == torch_col.Event.kColocateAdjustL2:
                    t0 = time.time()
                    old_gpu_mem = gpu_mem()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    hook.stub.adjust_l2_done()
                    t1 = time.time()
                    if not use_shared_tensor:
                        mem_info = f'gpu mem {old_gpu_mem:.1f} -> {gpu_mem():.1f}'
                    else:
                        mem_info = f'mem pool {torch_col.cuda_memory_pool_train_usage() / 1024 / 1024:.1f}M'
                    print('batch {} adjust : bs {} -> {} | {:.1f}ms | {:.1f}ms | {}'.format(
                        i, train_dataset.last_batch_size, train_dataset.batch_size, (time.time()-batch_begin) * 1000, (t1 - t0) * 1000, 
                        mem_info))
            train_dataset.next_batch()


        scheduler.step()
        end = time.time()
        if not use_shared_tensor:
            mem_info = f'memory {gpu_mem():.2f}Gb'
        else:
            mem_info = f'mem pool {torch_col.cuda_memory_pool_train_usage() / 1024 / 1024:.1f}M'
        batch_info = f'batch cnt {batch_cnt} avg {1e3*(end-begin)/batch_cnt:.1f}ms'
        if mode == 'task-switch-l1' or mode == 'colocate-l1':
            batch_info += f' | try {tried_batch} kill {killed_batch}, {killed_time*1e3:.1f}ms finish {finished_batch}, {finished_time*1e3:.1f}ms'
        print('[{} epoch {}] {:.3f}s | {} | batch-size {} | micro-batch-size {} | {} | wait_bs_valid {:.3f}s'.format(
                model.__class__.__name__, epoch, end - begin,
                batch_info, batch_size, train_dataset.batch_size,
                mem_info, wait_bs_valid_sec), flush=True)
    

    if mode == 'task-switch-l1' or mode == 'colocate-l1' or mode == 'colocate-l2':
        hook.stub.train_end()
        hook.stop()

    if mode == 'task-switch-l1' or mode == 'colocate-l1':
        print("[{}] Epoch x Batch {} | Batch Total Tried {} Killed {} Finished {}".format(
            model.__class__.__name__,
            num_epoch * ori_batch_size, total_tried_batch, total_killed_batch, total_finished_batch))


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Train Resnet')    
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-epoch', type=int, default=10)
    parser.add_argument('--mode', type=str, default='normal', 
                        choices=['normal', 
                                 'task-switch-l1', 'task-switch-l2','task-switch-l3', 
                                 'colocate-l1', 'colocate-l2'])
    # parser.add_argument('--dynamic-epoch-batch-size', action='store_true', default=False)
    # parser.add_argument('--dynamic-epoch-batch-size-schedule', nargs='*', type=int)
    args = parser.parse_args()
    
    batch_size = args.batch_size
    num_epoch = args.num_epoch

    print("resnet152 training, batch-size={}, num-epoch={}".format(batch_size, num_epoch))

    if 'task-switch-l1' in args.mode :
        hook = SwitchHook(args.mode)
        hook.stub.train_start()
    elif 'colocate-l1' in args.mode or 'colocate-l2' in args.mode:
        hook = ColocateHook(batch_size=batch_size)
        hook.stub.train_start()
    else:
        hook = None

    try:
        train(num_epoch=num_epoch, batch_size=batch_size,
              mode=args.mode, hook=hook)
    except SwitchL1Exception as e:
        print(e) # should not reach here

    # print(torch.cuda.memory_summary(), file=sys.stderr)

    torch_col.ReleaseMempool()
