import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, get_rank, get_world_size, destroy_process_group
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
import time

import torchvision
from torchvision.datasets import FakeData

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def prepare_dataloader(dataset:Dataset, batch_size: int):
    return DataLoader(
        dataset, 
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

def main(rank:int, world_size:int,
         num_epoch:int, batch_size:int):
    ddp_setup(rank, world_size)

    model = torchvision.models.swin_s()
    dataset = FakeData(size=1000, image_size=(3, 224, 224), num_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    dataloader = prepare_dataloader(dataset, batch_size)

    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.cuda(rank)

    # training
    model = DDP(model, device_ids=[rank])
    for e in range(num_epoch):
        epoch_begin = time.time()
        for b, (x, y) in enumerate(dataloader):
            x = x.cuda(rank, non_blocking=True)
            y = y.cuda(rank, non_blocking=True)
            
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            print(f'rank {rank} epoch {e} batch {b} loss {loss.item()}')
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_begin
        epoch_thpt = len(dataloader) * batch_size / epoch_time
        if rank == 0:
            print(f'[epoch {e}] thpt {epoch_thpt:.2f}')

    destroy_process_group()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, 10, 100), nprocs=world_size, join=True)
    