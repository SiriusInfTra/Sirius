import torch
from torch import nn
from torch.utils.data import DataLoader

from torchvision import models
from torchvision import transforms
from torchvision.datasets import FakeData

import numpy as np
import time
import argparse

# torch_col.disable_release_saved_tensor()

g_enable_cuda_sync = True

def cuda_sync():
    global g_enable_cuda_sync
    if not g_enable_cuda_sync:
        pass
    torch.cuda.synchronize()

class IntermMemoryStat:
    _model_param_ptr_set = set()
    _interm_memory_ptr_set = set()
    _nbytes_acc = 0
    
    @staticmethod
    def reset():
        IntermMemoryStat._interm_memory_ptr_set = set()
        IntermMemoryStat._nbytes_acc = 0

    @staticmethod
    def get():
        return IntermMemoryStat._nbytes_acc / 1024 / 1024 / 1024
    
    @staticmethod
    def add(x: torch.Tensor):
        if x.data_ptr() in IntermMemoryStat._interm_memory_ptr_set:
            return
        if x.data_ptr() in IntermMemoryStat._model_param_ptr_set:
            return
        IntermMemoryStat._interm_memory_ptr_set.add(x.data_ptr())
        IntermMemoryStat._nbytes_acc += x.element_size() * x.numel()

    @staticmethod
    def model_param_add(x: torch.Tensor):
        if x.data_ptr() in IntermMemoryStat._model_param_ptr_set:
            return
        IntermMemoryStat._model_param_ptr_set.add(x.data_ptr())


def register_saved_tensor_hook():
    def pack_hook(x):
        IntermMemoryStat.add(x)
        return x
    def unpack_hook(x):
        return x
    torch._C._autograd._push_saved_tensors_default_hooks(pack_hook, unpack_hook)


def gpu_memory_usage(device_id):
    free, total = torch.cuda.mem_get_info(device_id)
    return (total - free) / 1024 / 1024 / 1024


def allocated_memory(device_id):
    # return (total - free) / 1024 / 1024 / 1024
    # free, total = torch.cuda.mem_get_info(device_id)
    return torch.cuda.memory_reserved(device_id) / 1024 / 1024 / 1024

def train():
    # model = models.resnet152().cuda()
    model = models.vit_b_16().cuda()
    for param in model.parameters():
        IntermMemoryStat.model_param_add(param)

    criterion = nn.CrossEntropyLoss().cuda(0)
    optimizer = torch.optim.SGD(model.parameters(), 0.1, 
                                momentum=0.9, weight_decay=1e-4)

    eval_batch_size = [128, 64, 32, 16, 8, 4]
    # eval_batch_size = [128, ]
    epoch_micro_batch_size = [128] # warmup
    for bs in eval_batch_size:
        epoch_micro_batch_size.extend([bs] * 3)

    train_dataset = FakeData(128 * 10, (3, 224, 224), 50, transforms.ToTensor())
    data_loader = DataLoader(train_dataset, batch_size=128, 
                              shuffle=False, pin_memory=True, drop_last=False, num_workers=0)
    scaler = torch.cuda.amp.GradScaler()

    persist_memory = 0

    model.train()
    for epoch, mbs in enumerate(epoch_micro_batch_size):
        torch.cuda.reset_max_memory_allocated(0)
        interm_memorys = []
        compute_grad_times = []
        update_param_times = []
        batch_times = []

        # remove dataset prepare time
        xy = [(img, target) for img, target, in data_loader] 

        epoch_begin_time = time.time()
        for b, (images, targets) in enumerate(xy):
            batch_begin = time.time()

            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            optimizer.zero_grad()
            for mb in range(128 // mbs):
                IntermMemoryStat.reset()
                with torch.cuda.amp.autocast():
                    outputs = model(images[mb * mbs: (mb + 1) * mbs])
                    loss = criterion(outputs, targets[mb * mbs: (mb + 1) * mbs])
                scaler.scale(loss).backward()
                interm_memorys.append(IntermMemoryStat.get())
            cuda_sync()
            compute_grad_end = time.time()  

            scaler.step(optimizer)
            scaler.update()
            cuda_sync()
            update_param_end = time.time()

            batch_end = time.time()

            batch_times.append(batch_end - batch_begin)
            compute_grad_times.append(compute_grad_end - batch_begin)
            update_param_times.append(update_param_end - compute_grad_end)
        epoch_end_time = time.time()

        epoch_time = epoch_end_time - epoch_begin_time
        thpt = len(train_dataset) / epoch_time

        model_memory = 0
        optimizer_memory = 0
        # calculate model_memory and optimizer_memory here
        for param in model.parameters():
            model_memory += param.numel() * param.element_size()
        for buffer in model.buffers():
            model_memory += buffer.numel() * buffer.element_size()
        
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    optimizer_memory += v.numel() * v.element_size()

        persist_memory = (model_memory + optimizer_memory) / 1024 / 1024 / 1024

        if epoch > 0:
            actual_peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
            interm_memory = np.mean(interm_memorys)
            print(f'''Epoch {epoch} batch size {mbs} time: {epoch_time:.2f} thpt {thpt:.2f}
    GPU memory                  {gpu_memory_usage(0):.2f} GiB
    torch cached memory         {allocated_memory(0):.2f} GiB 
    actual peak memory          {actual_peak_memory:.2f} GiB
    interm memory               {interm_memory:.2f} GiB
    persist memory              {persist_memory:.2f} GiB
    current allocated memory    {torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024:.2f} GiB
    other memory                {actual_peak_memory - interm_memory - persist_memory:.2f} GiB

    batch time                  {np.mean(batch_times) * 1000:.2f} ms
    compute grad time           {np.mean(compute_grad_times) * 1000:.2f} ms
    update param time           {np.mean(update_param_times) * 1000:.2f} ms
''')
        else:
            print(f'warmup epoch {epoch} time: {epoch_time:.2f} thpt {thpt:.2f}')



    # print(f'persist memory {persist_memory}')

def main():
    parser = argparse.ArgumentParser('Train Resnet152')    
    args = parser.parse_args()
    register_saved_tensor_hook()
    train()


if __name__ == '__main__':
    main()