import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import models
from torchvision.datasets import FakeData
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Sampler
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
        

def train(num_epoch=10, batch_size=256, mode='normal', **kargs):
    if mode == 'task-switch-l1' or mode == 'colocate-l1' or mode == 'colocate-l2':
        hook = kargs["hook"]
    
    ori_batch_size = batch_size

    model = models.resnet101()
    model = model.cuda(0)

    criterion = nn.CrossEntropyLoss().cuda(0)
    optimizer = torch.optim.SGD(model.parameters(), 0.1, 
                                momentum=0.9, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # dummy data
    train_dataset = FakeData(1000, (3, 224, 224), 50, transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=False, pin_memory=True, drop_last=True, num_workers=1)
    
    if mode == 'task-switch-l1' or mode == 'colocate-l1':
        register_fbward_hook(model, hook.get_hook())
    print(f"train in {mode} mode")

    total_killed_batch = 0
    total_finished_batch = 0
    total_tried_batch = 0

    model.train()
    micro_batch_size = batch_size
    for epoch in range(num_epoch):
        begin = time.time()
        batch_cnt = 0
        killed_batch = 0
        finished_batch = 0
        tried_batch = 0
        wait_bs_valid_sec = 0 # add infer may cause batch size <= 0
        for i, (images, targets) in enumerate(train_loader):
            images:torch.Tensor
            targets:torch.Tensor
            images = images.to('cuda:0', non_blocking=True)
            targets = targets.to('cuda:0', non_blocking=True)
            
            # micro_batch_size = random.randint(1, batch_size)
            # print(micro_batch_size)
            batch_begin = time.time()
            while True:
                try:
                    # print('hook.cmd', hook.stub.cmd, flush=True)
                    if mode == 'task-switch-l1':
                        while hook.stub.cmd == torch_col.Event.kInterruptTrain:
                            time.sleep(1e-3)
                    if mode == 'colocate-l1' or mode == 'colocate-l2':
                        micro_batch_size = hook.stub.target_batch_size
                        wait_bs_valid_begin = time.time()
                        while micro_batch_size <= 0:
                            time.sleep(1e-3)
                            hook.try_reply_adjust_l1()
                            micro_batch_size = hook.stub.target_batch_size
                        wait_bs_valid_end = time.time()
                        wait_bs_valid_sec += wait_bs_valid_end - wait_bs_valid_begin
                    for i in range(0, batch_size, micro_batch_size):
                        tried_batch += 1
                        total_tried_batch += 1
                        output = model(images[i:i+micro_batch_size])
                        loss = criterion(output, targets[i:i+micro_batch_size]) / (batch_size / micro_batch_size)
                        # optimizer.zero_grad()
                        loss.backward()
                        finished_batch += 1
                        total_finished_batch += 1
                    optimizer.step()
                    optimizer.zero_grad()
                    batch_cnt += 1
                except (ColocateAdjustL1Exception, SwitchL1Exception) as e:
                    # micro_batch_size = int(micro_batch_size / 2 + 0.5)
                    killed_batch += 1
                    total_killed_batch += 1
                    micro_batch_size = hook.stub.target_batch_size
                    t0 = time.time()
                    optimizer.zero_grad()
                    torch.cuda.synchronize()
                    ori_gpu_mem = gpu_mem()
                    torch.cuda.empty_cache()
                    t1 = time.time()
                    if isinstance(e, ColocateAdjustL1Exception):
                        hook.stub.adjust_l1_done()
                    print('{} free cache {:.1f}ms, new micro batch size {}, gpu mem {:.1f} -> {:.1f}'.format(
                        e, (t1 - t0) * 1000, micro_batch_size, ori_gpu_mem, gpu_mem()))
                else:
                    if mode == 'colocate-l2' and hook.stub.cmd == torch_col.Event.kColocateAdjustL2:
                        t0 = time.time()
                        old_gpu_memory = gpu_mem()
                        old_batch_size = micro_batch_size
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        micro_batch_size = hook.stub.target_batch_size
                        hook.stub.adjust_l2_done()
                        t1 = time.time()
                        print('batch {} adjust : bs {} -> {} | {:.1f}ms | {:.1f}ms | gpu_mem {:.1f} -> {:.1f}'.format(
                            i, old_batch_size, micro_batch_size, (time.time()-batch_begin) * 1000, (t1 - t0) * 1000, 
                            old_gpu_memory, gpu_mem()))
                    break

        scheduler.step()
        end = time.time()
        # free, total = torch.cuda.mem_get_info(0)
        # used_mem = (total - free) / 1024 / 1024 / 1024
        used_mem = gpu_mem()
        batch_info = f'batch cnt {batch_cnt} avg {1e3*(end-begin)/batch_cnt:.1f}ms'
        if mode == 'task-switch-l1' or mode == 'colocate-l1':
            batch_info += f' | try {tried_batch} kill {killed_batch} finish {finished_batch}'
        print('[{} epoch {}] {:.3f}s | {} | batch-size {} | micro-batch-size {} | memory {:.2f}Gb | wait_bs_valid {:.3f}s'.format(
                model.__class__.__name__, epoch, end - begin,
                batch_info, batch_size, micro_batch_size,
                used_mem, wait_bs_valid_sec))
    
        # if hook.stub.cmd == torch_col.Event.kColocateAdjustL2:
        #     t0 = time.time()
        #     batch_size = int(batch_size - 2)
        #     train_loader = DataLoader(train_dataset, batch_size=batch_size, 
        #                               shuffle=False, pin_memory=True, drop_last=True, num_workers=1)
        #     torch.cuda.synchronize()
        #     torch.cuda.empty_cache()
        #     hook.stub.cmd = None
        #     hook.stub.adjust_l2_done()
        #     t1 = time.time()
        #     print('epoch adjust: bs {} | {:.1f}ms | gpu_mem {:.1f}'.format(
        #         batch_size, (t1 - t0) * 1000, gpu_mem()))

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

    # if args.dynamic_epoch_batch_size and len(args.dynamic_epoch_batch_size_schedule) > 0:
    #     batch_size = args.dynamic_epoch_batch_size_schedule[0]
    #     num_epoch = len(args.dynamic_epoch_batch_size_schedule)

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

    # print(torch.cuda.memory_summary())
