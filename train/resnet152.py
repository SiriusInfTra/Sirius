import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import models
from torchvision.datasets import FakeData
from torchvision import transforms
from torch.utils.data import DataLoader
import time
import argparse
import random
from multiprocessing.shared_memory import SharedMemory

def train(num_epoch=10, batch_size=256, accumulation_steps=1, test=False):
    batch_size //= accumulation_steps

    model = models.resnet101()
    model = model.cuda(0)

    criterion = nn.CrossEntropyLoss().cuda(0)
    optimizer = torch.optim.SGD(model.parameters(), 0.1, 
                                momentum=0.9, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # dummy data
    train_dataset = FakeData(100, (3, 224, 224), 50, transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=1,  pin_memory=True)

    model.train()
    for epoch in range(num_epoch):
        begin = time.time()
        batch_cnt = 0
        for i, (images, targets) in enumerate(train_loader):
            images:torch.Tensor
            targets:torch.Tensor
            images = images.to('cuda:0', non_blocking=True)
            targets = targets.to('cuda:0', non_blocking=True)

            output = model(images)
            loss = criterion(output, targets) / accumulation_steps

            optimizer.zero_grad()
            loss.backward()
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            batch_cnt += 1

        scheduler.step()
        end = time.time()
        free, total = torch.cuda.mem_get_info(0)
        used_mem = (total - free) / 1024 / 1024 / 1024
        print('[{} epoch {}] {:.3f}s | mini-batch {:.1f}ms | memory {:.2f}Gb'.format(
                model.__class__.__name__, epoch, 
                end - begin, 
                1000*(end-begin)/batch_cnt,
                used_mem))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train Resnet')    
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-epoch', type=int, default=10)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    print("resnet152 training, batch-size={}, num-epoch={}".format(batch_size, num_epoch))

    train(num_epoch=num_epoch, batch_size=batch_size, accumulation_steps=1, test=args.test)