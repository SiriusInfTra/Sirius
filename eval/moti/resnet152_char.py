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

def get_gpu_memory(device_id):
    free, total = torch.cuda.mem_get_info(device_id)
    return (total - free) / 1024 / 1024 / 1024

def train():
    model = models.resnet152().cuda()

    criterion = nn.CrossEntropyLoss().cuda(0)
    optimizer = torch.optim.SGD(model.parameters(), 0.1, 
                                momentum=0.9, weight_decay=1e-4)

    eval_batch_size = [128, 64, 32, 16, 8, 4]
    epoch_micro_batch_size = [128] # warmup
    for bs in eval_batch_size:
        epoch_micro_batch_size.extend([bs] * 3)

    train_dataset = FakeData(128 * 10, (3, 224, 224), 50, transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=128, 
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
        solo_step_times = []

        epoch_begin_time = time.time()
        for b, (images, targets) in enumerate(train_loader):
            batch_begin = time.time()

            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            optimizer.zero_grad()
            for mb in range(128 // mbs):
                with torch.cuda.amp.autocast():
                    outputs = model(images[mb * mbs: (mb + 1) * mbs])
                    loss = criterion(outputs, targets[mb * mbs: (mb + 1) * mbs])
                scaler.scale(loss).backward()
            torch.cuda.synchronize()
            compute_grad_end = time.time()

            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            update_param_end = time.time()

            batch_end = time.time()

            batch_times.append(batch_end - batch_begin)
            compute_grad_times.append(compute_grad_end - batch_begin)
            update_param_times.append(update_param_end - compute_grad_end)
        epoch_end_time = time.time()

        epoch_time = epoch_end_time - epoch_begin_time
        thpt = len(train_dataset) / epoch_time

        if epoch > 0:
            actual_peak_memory = torch.cuda.max_memory_allocated()
            interm_memory = np.mean(interm_memorys) / 1024 / 1024 / 1024
            print(f'''Epoch {epoch} time: {epoch_time} thpt {thpt}
    GPU memory {get_gpu_memory(0)} 
    actual peak memory {actual_peak_memory} 
    interm memory {interm_memory}
    persist memory {persist_memory}
    current allocated memory {torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024}
    other memory {actual_peak_memory - interm_memory - persist_memory}
''')
        else:
            print(f'warmup epoch {epoch} time: {epoch_time} thpt {thpt}')
            
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

            persist_memory = model_memory + optimizer_memory



    # print(f'persist memory {persist_memory}')

def main():
    parser = argparse.ArgumentParser('Train Resnet152')    
    args = parser.parse_args()
    train()


if __name__ == '__main__':
    main()