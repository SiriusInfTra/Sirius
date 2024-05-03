import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, get_rank, get_world_size, destroy_process_group
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler
import os
import time

import torchvision
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor


# class MyDataset(Dataset):
#     def __init__(self) -> None:
#         super().__init__()
#         self.size = 1000
#         self.data = torch.randn(1000, 3, 224, 224).pin_memory()
#         self.label = torch.randint(0, 10, (1000,)).pin_memory()

#     def __len__(self):
#         return self.size
    
#     def __getitem__(self, index):
#         return self.data[index], self.label[index]

class MyIterDataset(IterableDataset):
    def __init__(self, batch_size):
        super().__init__()
        self.size = 1000
        self.data = torch.randn(1000, 3, 224, 224).pin_memory()
        self.label = torch.randint(0, 10, (1000,)).pin_memory()
        self.index = 0
        self.b_sz = batch_size

    def __iter__(self):
        while True:
            if self.index >= self.size:
                self.index = 0
                break
            cur_b_sz = min(self.b_sz, self.size - self.index)
            data, label = self.data[self.index:self.index+cur_b_sz], self.label[self.index:self.index+cur_b_sz]
            # print(data.shape, label.shape)
            self.index += cur_b_sz
            yield data, label
    
    def __len__(self):
        return self.size


def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def prepare_dataloader(batch_size: int):
    dataset = MyIterDataset(batch_size)
    dataloader = DataLoader(
        dataset, 
        batch_size=None,
        pin_memory=False,
        shuffle=None,
        num_workers=0
        # sampler=DistributedSampler(dataset)
    )
    return dataset, dataloader


def main(rank:int, world_size:int,
         num_epoch:int, batch_size:int):
    ddp_setup(rank, world_size)

    # dataset = FakeData(size=1000, image_size=(3, 224, 224), num_classes=10, transform=ToTensor())
    # dataset = {'image': torch.randn(1000, 3, 224, 224), 'label': torch.randint(0, 10, (1000))}
    # dataset = MyDataset()

    
    model = torchvision.models.swin_t()
    model = model.cuda(rank)
    model = DDP(model, device_ids=[rank], output_device=rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    dataset, dataloader = prepare_dataloader(batch_size)

    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.cuda(rank)

    model.train()

    # training
    print(f'rank {rank}')
    for e in range(num_epoch):
        epoch_begin = time.time()
        # dataloader.sampler.set_epoch(e)
        for b, (x, y) in enumerate(dataloader):
            x = x.cuda(rank, non_blocking=True)
            y = y.cuda(rank, non_blocking=True)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            # print(f'rank {rank} epoch {e} batch {b} loss {loss.item()}')
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_begin
        epoch_thpt = len(dataset) / epoch_time
        if rank == 0:
            print(f'[epoch {e}] {epoch_time:.2f}s thpt {epoch_thpt:.2f}')

    destroy_process_group()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, 10, 100), nprocs=world_size, join=True)
    