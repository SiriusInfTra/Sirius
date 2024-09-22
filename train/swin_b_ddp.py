import contextlib
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as torch_dist
import torch.multiprocessing as torch_mp
from torch import nn
import swin_b_tiny
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os, sys
import torch_col
from torch_col import MemoryPool, EventManager, TrainMode, HookMode
from torch_col import DynamicBatchDataset
import torch_col.trainer
import torch_col.xsched
from typing import Optional, cast
import time
from torch.utils.data import Dataset
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset


def setup(rank, world_size):
    cuda_index = torch.cuda._get_nvml_device_index(0)
    torch_col.info(f'[Train rank {rank} PID {os.getpid()} world size {world_size}] setup')
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(22345 + cuda_index)
    torch_col.setup_colocate_training(rank, world_size, True, True)

def cleanup():
    torch_col.cleanup_colocate_training(True)


def train(rank:int, world_size:int,
          train_mode: TrainMode, 
          num_epoch: int, real_data: bool, batch_size: int, 
          global_batch_size: Optional[int] = None):
    setup(rank, world_size)
    torch_col.init_train_info(batch_size, batch_size, model_name='swin_b_ddp')
    print(f'Env: {os.environ}')
    print(f'Real Data: {real_data}')

    hook_mode = torch_col.get_hook_mode()
    
    if torch_col.is_enable_shared_tensor():
        torch_col.tag_model_start()


    model = swin_b_tiny.SwinTransformer(
        input_shape=(32, 32, 3),
        patch_size=(2, 2),
        qkv_bias=True,
        num_classes=100,
        embed_dim=128,
        num_heads=8,
        num_mlp=256,
        window_size=2,
        shift_size=1,
        dropout_rate=0.1
    ).cuda()

    # for param in model.parameters():
    #     param.requires_grad = False
    if os.environ.get('COL_IMAGENET', '0') == '1':
        num_class = 100
    elif os.environ.get('COL_CIFAR100', '0') == '1':
        num_class = 100
    elif real_data:
        num_class = 37
    else:
        num_class = 10
    print(f"num_class = {num_class}.")
    if real_data:
        model.head = torch.nn.Linear(model.head.in_features, num_class).cuda()
    model = DDP(model, device_ids=[rank])

    print(f"Train params memory usage: {torch_col.MemoryPool.get_memory_usage() * 1024:.2f}M")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).cuda(rank)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    # optimizer, scheduler = torch_col.get_imagenet_optimizer_and_scheduler(model, global_batch_size, 90, 5)
    scaler = torch.cuda.amp.GradScaler()

    print(f"Train after init memory pool usage: {MemoryPool.get_memory_usage() * 1024:.2f}M")
    
    hook = torch_col.get_hook(train_mode, hook_mode, num_epoch, batch_size)
    hook.register_pytorch_hook([model, criterion])
    # checkpoint_micro_batch = hook.train_mode.is_kill_batch()
    checkpoint_micro_batch = True

    # dummy data
    train_dataset = DynamicBatchDataset(
        model_name='swin_b', size=3680 if real_data else 1000, max_batch_size=batch_size, 
        hook=hook, trace=None,
        input_shape=(3, 32, 32), num_class=num_class,
        fake_data=not real_data,
        max_global_batch_size=global_batch_size,
        checkpoint_micro_batch=checkpoint_micro_batch)

    model.train()
    hook.train_start()
    
    transform0 = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])

    # torch_col.util.initialize_sgd_optimizer(model, optimizer)
    if train_dataset.checkpoint_micro_batch:
        grad_accumulator = torch_col.GradAccumulator(model)
    else:
        grad_accumulator = None

    print('swin after initialize, allocated {} cached {}'.format(
        torch_col.MemoryPool.get_allocated_memory(), 
        torch_col.MemoryPool.get_memory_usage()),
        flush=True, file=sys.stderr)

    torch_col.dist.wait_barrier()

    def iter_train_fn(batch):
        images = batch['image']
        targets = batch['label']
        images: torch.Tensor = images.cuda(rank, non_blocking=True)
        images = transform0(images)
        targets: torch.Tensor = targets.cuda(rank, non_blocking=True)
        with train_dataset.get_context(model):
            # if grad_accumulator is not None:
            optimizer.zero_grad(set_to_none=False)
            with torch.cuda.amp.autocast(cache_enabled=False):
                output: torch.Tensor = model(images)
                loss = criterion(output, targets)
                train_dataset.scale_loss(loss)
                # torch_col.info(f"Batch Size is {len(images)}.")
            scaler.scale(loss).backward()    
        train_dataset.step_stage(hook, optimizer, scaler=scaler, 
                                 grad_accumulator=grad_accumulator)
        return loss

    trainer = torch_col.Trainer(model, train_dataset, iter_train_fn)
    if os.environ.get('COL_IMAGENET', '0') == '1':
        import numpy as np
        val_images = np.load('imagenet-100/imagenet-100-images-val.npy', mmap_mode='c')
        val_labels = np.load('imagenet-100/imagenet-100-labels-val.npy', mmap_mode='c')
        val_dataset = TensorDataset(torch.tensor(val_images, dtype=torch.float32), torch.tensor(val_labels, dtype=torch.long))
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    elif os.environ.get('COL_CIFAR100', '0') == '1':
        # 下载并加载 CIFAR-100 验证数据集
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        print('==> Preparing data..')
        # 加载验证数据集
        val_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

        # 提取图像和标签数据
        val_images = []
        val_labels = []

        for image, label in val_dataset:
            val_images.append(image)
            val_labels.append(label)

        # 将列表转换为张量
        val_images = torch.stack(val_images)
        val_labels = torch.tensor(val_labels, dtype=torch.long)

        # 创建 TensorDataset 和 DataLoader
        val_dataset = TensorDataset(val_images, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    for epoch in range(num_epoch):
        last_loss = trainer.train_one_epoch(epoch)
        lr = optimizer.param_groups[0]['lr']
        # scheduler.step(epoch)   
        # scheduler.step(epoch)
        epoch_stat = trainer.get_last_epoch_stat()
        epoch_duration = trainer.get_last_epoch_duration()

        mem_info = f'mem {MemoryPool.get_memory_usage():.2f}Gb | {MemoryPool.get_memory_peak_usage():.2f}Gb | {MemoryPool.get_allocated_memory():.2f}Gb'
        batch_info = (
            f'batch cnt {epoch_stat.finished_batch} ' 
            + f'avg {epoch_duration/epoch_stat.finished_batch:.1f}ms')
        if train_mode.is_kill_batch():
            batch_info += (
                f' | try {epoch_stat.tried_batch} kill {epoch_stat.killed_batch}, ' 
                + f'{epoch_stat.killed_time*1e3:.1f}ms finish {epoch_stat.finished_batch}, '
                + f'{epoch_stat.finished_time*1e3:.1f}ms')
        # time.sleep(10)
        # torch.cuda.synchronize()
        # torch_col.dist.wait_barrier()
        # torch_col.info('Before Swin Buffer1', cast(swin_b_tiny.SwinTransformer, model.module).swin_block1.attn.get_buffer('relative_position_index').cpu())
        # torch_col.info('Before Swin Buffer2', cast(swin_b_tiny.SwinTransformer, model.module).swin_block2.attn.get_buffer('relative_position_index').cpu())
        if os.environ.get('COL_IMAGENET', '0') == '1' or os.environ.get('COL_CIFAR100', '0') == '1':
            while True:
                total_loss = 0.0
                correct = 0
                try:
                    # with contextlib.nullcontext():
                    with hook.steps_no_interrupt():
                        with torch.no_grad():
                            for images, labels in val_loader:
                                images = torch.tensor(images).cuda()
                                labels = torch.tensor(labels).cuda()
                                outputs: torch.Tensor = model.module(images)
                                loss = criterion(outputs, labels)
                                # torch.cuda.synchronize()
                                correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
                                total_loss += loss.item()
                except torch_col.ColocateAdjustL1Exception as e:
                    # not calculate success retry
                    print(f'Exception in val : {e}, retry.')
                    hook.release_and_reply()
                    continue
                break
            torch_col.info('After Swin Buffer1', cast(swin_b_tiny.SwinTransformer, model.module).swin_block1.attn.get_buffer('relative_position_index').cpu())
            torch_col.info('After Swin Buffer2', cast(swin_b_tiny.SwinTransformer, model.module).swin_block2.attn.get_buffer('relative_position_index').cpu())
                # torch_col.dist.wait_barrier()
            val_loss = total_loss / len(val_loader)
            val_acc = correct / len(val_loader.dataset)
            batch_info += f' | val loss {val_loss:.6f} | val acc {val_acc:.6f}'
        # torch_col.dist.wait_barrier()
        if global_batch_size is not None:
            batch_info += f' | num_rollback_sampels {train_dataset.num_rollback_samples_in_epoch}'
        print('[Rank {} | {} epoch {}] {:.3f}s | {} | batch-size {} | micro-batch-size {} | {} | thpt {:.2f} | lr {:.2e} | loss {:.6f}'.format(
                rank, model.__class__.__name__, epoch, epoch_duration / 1e3,
                batch_info, batch_size, train_dataset.batch_size,
                mem_info, epoch_stat.finished_sample / (epoch_duration / 1e3), 
                lr,
                last_loss.item()), flush=True)
        if real_data and torch_col.get_train_rank() == 0:
            # pass
            
            checkpoint_dir = os.environ.get('COL_CP', None)
            
            if checkpoint_dir is not None:
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                torch.save(model.module.state_dict(), f'{checkpoint_dir}/{os.getpid()}_{epoch}.pth')
    
        if hook.can_exit_after_infer_worklaod_done():
            print('[hook] inference workload done, will exit training', flush=True)
            torch_col.dist.wait_barrier()
            break

    if train_mode.is_kill_batch():
        overall_stat = trainer.get_overall_stat()
        print("[{}] Epoch x Batch {} | Batch Total Tried {} Killed {} Finished {}".format(
            model.__class__.__name__,
            num_epoch * batch_size, overall_stat.tried_batch, 
            overall_stat.killed_batch, overall_stat.finished_batch), flush=True)

    hook.train_end()
    hook.stop()

    EventManager.dump(None, train_mode)
    cleanup()


def main():
    parser = argparse.ArgumentParser('Train Resnet')    
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--global-batch-size', type=int, default=512)
    parser.add_argument('--num-epoch', type=int, default=15)
    parser.add_argument('--train-mode', type=str, default=TrainMode.COLOCATE_L1.value, 
                        choices=[train_mode.value for train_mode in TrainMode])
    parser.add_argument('--real-data', action='store_true')
    args = parser.parse_args()
    
    batch_size = args.batch_size
    global_batch_size = args.global_batch_size
    num_epoch = args.num_epoch
    num_epoch = 90
    real_data = args.real_data or True
    train_mode = [train_mode for train_mode in TrainMode if train_mode.value == args.train_mode][0]
    # hook_mode = [hook_mode for hook_mode in HookMode if hook_mode.value == args.hook_mode][0]
    # hook_mode = torch_col.get_hook_mode()
    
    
    print(f"Swin Transformer training, batch-size={batch_size}"
        + f", num-epoch={num_epoch}, train-mode={train_mode}")
    
    process_context =  torch_mp.spawn(train, 
                   args=(torch.cuda.device_count(), train_mode,
                         num_epoch, real_data, batch_size, global_batch_size), 
                   nprocs=torch.cuda.device_count(), 
                   join=False)
    try:
        process_context.join()
    except Exception as e:
        torch_col.info(f"Exception in training: {e}")
        for p in process_context.processes:
            p.kill()
            p.join()
        os.abort()


if __name__ == '__main__':
    main()