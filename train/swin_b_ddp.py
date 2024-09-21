import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as torch_dist
import torch.multiprocessing as torch_mp
from torch import nn
from torchvision import models
from torch.utils.data import DataLoader
import argparse
import os, sys

import torch_col
from torch_col import MemoryPool, EventManager, TrainMode, ColocateCtrlHookMode
from torch_col import DynamicBatchDataset
from torch_col import dynamic_batch
import torch_col.trainer
import torch_col.xsched
from typing import Optional
import time


def setup(rank, world_size):
    torch_col.info(f'[Train rank {rank} PID {os.getpid()} world size {world_size}] setup')
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(12345 + os.getuid() % 100)
    torch_col.setup_colocate_training(rank, world_size, True, True)

def cleanup():
    torch_col.cleanup_colocate_training(True)


def train(rank:int, world_size:int,
          num_epoch: int, batch_size: int, 
          global_batch_size: Optional[int] = None):
    setup(rank, world_size)
    torch_col.init_train_info(batch_size, batch_size, 
                              model_name='swin_b_ddp')

    hook_mode = torch_col.get_colocate_ctrl_hook_mode()
    train_mode = torch_col.get_colocate_train_mode()
    
    if torch_col.is_enable_shared_tensor():
        torch_col.tag_model_start()

    model = models.swin_b(weights=models.Swin_B_Weights.DEFAULT).cuda()
    model = DDP(model, device_ids=[rank])

    print(f"Train params memory usage: {torch_col.MemoryPool.get_memory_usage() * 1024:.2f}M")

    criterion = nn.CrossEntropyLoss().cuda(rank)
    optimizer = torch.optim.SGD(model.parameters(), 0.001, 
                                momentum=0.9, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    print(f"Train after init memory pool usage: {MemoryPool.get_memory_usage() * 1024:.2f}M")
    
    col_ctrl = torch_col.create_colocate_ctrl(train_mode, hook_mode, num_epoch, batch_size)
    col_ctrl.register_pytorch_hook([model, criterion])

    # dummy data
    # train_dataset = DynamicBatchDataset(
    #     model_name='swin_b', size=1000, max_batch_size=batch_size, 
    #     hook=hook, trace=None,
    #     input_shape=(3, 224, 224), num_class=10, 
    #     max_global_batch_size=None,
    #     checkpoint_micro_batch=checkpoint_micro_batch)

    enable_grad_accumulate = col_ctrl.train_mode.is_colocate()
    checkpoint_micro_batch = col_ctrl.train_mode.is_kill_batch()
    dataset, batch_manager = torch_col.init_dynamic_batch(
        dataset_size=1000 * world_size, 
        dataset_type=dynamic_batch.DatasetType.VISION,
        dataset_config=dynamic_batch.VisionDatasetConfig((3, 224, 224), 10),
        batch_size=batch_size,
        global_batch_size=None,
        enable_grad_accumulate=enable_grad_accumulate,
        checkpoint_micro_batch=checkpoint_micro_batch,
    )   

    model.train()
    col_ctrl.train_start()

    torch_col.util.initialize_sgd_optimizer(model, optimizer)
    if batch_manager._checkpoint_micro_batch:
        grad_accumulator = torch_col.GradAccumulator(model)
    else:
        grad_accumulator = None

    print('swin after initialize, allocated {} cached {}'.format(
        torch_col.MemoryPool.get_allocated_memory(), 
        torch_col.MemoryPool.get_memory_usage()),
        flush=True, file=sys.stderr)

    torch_col.dist.wait_barrier()

    def iter_train_fn(batch: dynamic_batch.Batch):
        images = batch['images']
        targets = batch['labels']
        images: torch.Tensor = images.cuda(rank, non_blocking=True)
        targets: torch.Tensor = targets.cuda(rank, non_blocking=True)
        with batch_manager.ddp_sync_context(model, batch):
            with torch.cuda.amp.autocast(cache_enabled=False):
                output = model(images)
                loss = criterion(output, targets)
                running_loss = loss.item() * images.size(0)
                batch_manager.scale_loss(batch, loss)
            scaler.scale(loss).backward()
        batch_manager.optimizer_step(batch, optimizer, 
                                     amp_scaler=scaler, 
                                     grad_accumulator=grad_accumulator)
        return running_loss

    trainer = torch_col.Trainer(model, iter_train_fn)

    for epoch in range(num_epoch):
        running_loss = trainer.train_one_epoch(epoch)

        epoch_stat = trainer.get_last_epoch_stat()
        epoch_duration = trainer.get_last_epoch_duration()

        mem_info = f'mem {MemoryPool.get_memory_usage():.2f}Gb'
        batch_info = (
            f'batch cnt {epoch_stat.finished_batch} ' 
            + f'avg {epoch_duration/epoch_stat.finished_batch:.1f}ms')
        if train_mode.is_kill_batch():
            batch_info += (
                f' | try {epoch_stat.tried_batch} kill {epoch_stat.killed_batch}, ' 
                f'{epoch_stat.killed_time*1e3:.1f}ms finish {epoch_stat.finished_batch}, '
                f'{epoch_stat.finished_time*1e3:.1f}ms')
        if global_batch_size is not None:
            batch_info += f' | num_rollback_sampels {epoch_stat.num_rollback_batch}'

        print(f'[Rank {rank} | {model.__class__.__name__} epoch {epoch}] '
              f'{epoch_duration / 1e3:.3f}s | {batch_info} | batch-size {batch_size} '
              f'| {mem_info} | thpt {epoch_stat.finished_sample / (epoch_duration / 1e3):.3f} '
              f'| loss {running_loss:.6f}', 
              flush=True)
    
        if col_ctrl.can_exit_after_infer_worklaod_done():
            print('[hook] inference workload done, will exit training', flush=True)
            torch_col.dist.wait_barrier()
            break

    if train_mode.is_kill_batch():
        overall_stat = trainer.get_overall_stat()
        print("[{}] Epoch x Batch {} | Batch Total Tried {} Killed {} Finished {}".format(
            model.__class__.__name__,
            num_epoch * batch_size, overall_stat.tried_batch, 
            overall_stat.killed_batch, overall_stat.finished_batch), flush=True)

    col_ctrl.train_end()
    col_ctrl.stop()

    EventManager.dump(None, train_mode)
    cleanup()


def main():
    parser = argparse.ArgumentParser('Train Resnet')    
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--global-batch-size', type=int, default=500)
    parser.add_argument('--num-epoch', type=int, default=15)
    args = parser.parse_args()
    
    batch_size = args.batch_size
    global_batch_size = args.global_batch_size
    num_epoch = args.num_epoch

    print(f"Swin Transformer training, batch-size={batch_size}"
          f", num-epoch={num_epoch}")
    
    process_context =  torch_mp.spawn(train, 
                   args=(torch.cuda.device_count(),
                         num_epoch, batch_size, global_batch_size), 
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
    # torch_col.util.cleanup_previous_shm()
    main()