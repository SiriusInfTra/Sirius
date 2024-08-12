import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as torch_dist
import torch.multiprocessing as torch_mp
from torch import nn
import torch_col.xsched
from torchvision import models
from torch.utils.data import DataLoader
import argparse
import os, sys

import torch_col
from torch_col import MemoryPool, EventManager, TrainMode, HookMode
from torch_col import ColocateAdjustL1Exception, SwitchL1Exception
from torch_col import CustomeDynamicBatchDataset
from typing import Optional
import time


def setup(rank, world_size):
    print(f'[Train rank {rank} PID {os.getpid()} world size {world_size}] setup')
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(12345 + os.getuid() % 100)
    torch_col.torch_col_init(rank, world_size)
    torch_col.xsched.guess_nccl_begin()

    torch.cuda.set_device(rank)
    stream = torch.cuda.Stream(device=rank)

    if torch_col.is_enable_xsched():
        torch_col.xsched.register_stream(stream)
        print("CUDA Stream create with xsched registered.")
    else:
        print(f"CUDA Stream create without xsched.")
    torch.cuda.set_stream(stream)

    torch_dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    torch_dist.GroupMember.WORLD._get_backend(torch.device('cuda'))._set_as_default_pg()

    import time
    time.sleep(10)

def cleanup():
    if torch_col.is_enable_xsched():
        torch_col.xsched.unregister_all_streams()
    torch_dist.destroy_process_group()


def train(rank:int, world_size:int,
          train_mode: TrainMode, 
          num_epoch: int, batch_size: int, 
          global_batch_size: Optional[int] = None):
    setup(rank, world_size)
    gloo_group = torch_dist.new_group(backend='gloo')

    # batch_size = 32 if rank == 0 else 36

    hook_mode = torch_col.get_hook_mode()
    
    if torch_col.is_enable_shared_tensor():
        torch_col.tag_model_start()


    model = models.swin_b(weights=models.Swin_B_Weights.DEFAULT).cuda()
    model = DDP(model, device_ids=[rank])

    print(f"Train params memory usage: {torch_col.MemoryPool.get_memory_usage() * 1024:.2f}M")

    criterion = nn.CrossEntropyLoss().cuda(rank)
    optimizer = torch.optim.SGD(model.parameters(), 0.1, 
                                momentum=0.9, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    print(f"Train after init memory pool usage: {MemoryPool.get_memory_usage() * 1024:.2f}M")
    
    hook = torch_col.get_hook(train_mode, hook_mode, num_epoch, batch_size)
    hook.register_pytorch_hook([model, criterion])
    checkpoint_micro_batch = hook.train_mode.is_kill_batch()
    # checkpoint_micro_batch = False

    # dummy data
    train_dataset = CustomeDynamicBatchDataset(
        model_name='swin_b', size=1000, max_batch_size=batch_size, 
        hook=hook, trace=None,
        input_shape=(3, 224, 224), num_class=10, 
        max_global_batch_size=None,
        checkpoint_micro_batch=checkpoint_micro_batch)
    train_loader = DataLoader(train_dataset, batch_size=None, 
                              shuffle=False, pin_memory=True, drop_last=False, num_workers=0)

    total_killed_batch = 0
    total_finished_batch = 0
    total_tried_batch = 0

    model.train()
    hook.train_start()

    # torch_dist.barrier(None)
    # torch_dist.barrier(group=gloo_group)

    torch_col.util.initialize_sgd_optimizer(model, optimizer)
    if train_dataset.checkpoint_micro_batch:
        # grad_accumulator = torch_col.GradAccumulator(model)
        grad_accumulator = None
    else:
        grad_accumulator = None

    print('swin after initialize, allocated {} cached {}'.format(
        torch_col.MemoryPool.get_allocated_memory(), 
        torch_col.MemoryPool.get_memory_usage()),
        flush=True, file=sys.stderr)

    # print_opt(optimizer)

    gloo_group.barrier()


    for epoch in range(num_epoch):
        epoch_event = EventManager.record_event(f'epoch_{epoch:02d}_{train_dataset.size}')
        batch_cnt = 0
        killed_batch = 0
        finished_batch = 0
        tried_batch = 0
        killed_time = 0
        finished_time = 0
        wait_bs_valid_sec = 0 # add infer may cause batch size <= 0
        finished_imgs = 0
        for i, batch in enumerate(train_loader):
            # torch_dist.barrier(group=gloo_group)
            # print(f'[epoch {epoch} batch {i}] batch size {len(images)}', flush=True, file=sys.stderr)
            images = batch['image']
            targets = batch['label']
            train_dataset.record_batch_event(epoch, i, len(images), train_dataset.global_batch_size)
            images: torch.Tensor = images.cuda(rank, non_blocking=True)
            targets: torch.Tensor = targets.cuda(rank, non_blocking=True)
            gloo_group.barrier()
            try:
                tried_batch += 1
                total_tried_batch += 1
                optimizer.zero_grad(set_to_none=False)
                with torch.cuda.amp.autocast(cache_enabled=False):
                    output = model(images)
                    loss = criterion(output, targets)
                    # train_dataset.scale_loss(loss)
                scaler.scale(loss).backward()
                train_dataset.step_stage(hook, optimizer, scaler=scaler, 
                                         grad_accumulator=grad_accumulator)

                finished_batch += 1
                total_finished_batch += 1
                batch_cnt += 1
                finished_imgs += len(images)

            except (ColocateAdjustL1Exception, SwitchL1Exception, torch_col.EngineColocateAdjustL1Exception) as e:
                if epoch == 0 and i == 0:
                    raise RuntimeError("micro batch 0 could not be interrupted.")
                # for fast develop, should merge to hook.py
                torch_col.info(f'epoch {epoch} batch {i} drop')
                gloo_group.barrier()
                # torch_dist.GroupMember.WORLD._get_backend(torch.device('cuda'))._restart_nccl_comm([torch.device(f'cuda:{rank}')])
                if isinstance(e, torch_col.EngineColocateAdjustL1Exception):
                    torch_col.xsched.kill_batch()

                torch.cuda.synchronize()
                model.reducer.finalize_dropped_batch()
                gloo_group.barrier()
                killed_batch += 1
                total_killed_batch += 1
                train_dataset.cancel_micro_batch()
                with EventManager.record_duration_event(f'batch_exception_{epoch:02d}_{i:03d}_{len(images):02d}'):
                    # cuda has alreadly synced
                    hook.release_and_reply()
                    print(f'[{e}] batch_size: {len(images)} -> {train_dataset.batch_size}.')

                # gloo_group.barrier()
                torch_dist.barrier(gloo_group)
                torch_dist.GroupMember.WORLD._get_backend(torch.device('cuda'))._restart_nccl_comm([torch.device(f'cuda:{rank}')])


                print(f'[PID {os.getpid()}] sync training begin', flush=True, file=sys.stderr)
                torch.cuda.synchronize()
                # torch_dist.barrier(group=gloo_group)
                print(f'[PID {os.getpid()}] sync training done', flush=True, file=sys.stderr)
                # import time
                # time.sleep(30)

                # tmp_ts = torch.tensor([-1] * 100, dtype=int, device='cuda')
                # work = torch_dist.broadcast(tmp_ts, 0)
                # torch_col.info(f'tmp_ts {tmp_ts}')
            else:
                # torch.cuda.current_stream().synchronize()
                train_dataset.next_batch()
                if epoch == 0 and i == 0:
                    if torch_col.is_enable_shared_tensor():
                        torch_col.tag_model_end()
            if torch_col.is_enable_xsched():
                if epoch == 0 and i == 0:
                    torch_col.xsched.guess_nccl_end()
                    nccl_streams = torch_col.xsched.get_nccl_streams()
                    assert len(nccl_streams) <= 1, f'nccl streams {nccl_streams}'
                    if len(nccl_streams) == 1:
                        torch_col.xsched.register_stream(nccl_streams[0], False)
                torch_col.xsched.initial_kill_batch(epoch, i)
            torch.cuda.synchronize()

        EventManager.record_event('', epoch_event)

        mem_info = f'mem {MemoryPool.get_memory_usage():.2f}Gb'
        batch_info = f'batch cnt {batch_cnt} avg {epoch_event.duration/batch_cnt:.1f}ms'
        if train_mode.is_kill_batch():
            batch_info += f' | try {tried_batch} kill {killed_batch}, {killed_time*1e3:.1f}ms finish {finished_batch}, {finished_time*1e3:.1f}ms'
        if global_batch_size is not None:
            batch_info += f' | num_rollback_sampels {train_dataset.num_rollback_samples_in_epoch}'
        print('[Rank {} | {} epoch {}] {:.3f}s | {} | batch-size {} | micro-batch-size {} | {} | thpt {:.2f} | wait_bs_valid {:.3f}s | loss {:.6f}'.format(
                rank, model.__class__.__name__, epoch, epoch_event.duration / 1e3,
                batch_info, batch_size, train_dataset.batch_size,
                mem_info, finished_imgs / (epoch_event.duration / 1e3), wait_bs_valid_sec, loss.item()), flush=True)
    
        if hook.can_exit_after_infer_worklaod_done():
            print('[hook] inference workload done, will exit training', flush=True)
            break

    if train_mode.is_kill_batch():
        print("[{}] Epoch x Batch {} | Batch Total Tried {} Killed {} Finished {}".format(
            model.__class__.__name__,
            num_epoch * batch_size, total_tried_batch, total_killed_batch, total_finished_batch))

    hook.train_end()
    hook.stop()

    EventManager.dump(None, train_mode)
    cleanup()

def main():
    parser = argparse.ArgumentParser('Train Resnet')    
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--global-batch-size', type=int, default=500)
    parser.add_argument('--num-epoch', type=int, default=15)
    parser.add_argument('--train-mode', type=str, default=TrainMode.COLOCATE_L1.value, 
                        choices=[train_mode.value for train_mode in TrainMode])
    parser.add_argument('--train-profile', type=str, default='train-profile.csv')
    # parser.add_argument('--hook-mode', default=HookMode.XSCHED_SYNC.value, 
    #                     choices=[hook_mode.value for hook_mode in HookMode])
    # parser.add_argument('--use-xsched', type=bool) # not used args
    args = parser.parse_args()
    
    batch_size = args.batch_size
    global_batch_size = args.global_batch_size
    num_epoch = args.num_epoch
    train_mode = [train_mode for train_mode in TrainMode if train_mode.value == args.train_mode][0]
    # hook_mode = [hook_mode for hook_mode in HookMode if hook_mode.value == args.hook_mode][0]
    # hook_mode = torch_col.get_hook_mode()
    
    
    print(f"Swin Transformer training, batch-size={batch_size}"
        + f", num-epoch={num_epoch}, train-mode={train_mode}")
    torch_mp.spawn(train, 
                   args=(torch.cuda.device_count(),train_mode,
                         num_epoch, batch_size, global_batch_size), 
                   nprocs=torch.cuda.device_count(), 
                   join=True)
    

    # stream = torch.cuda.Stream()
    # # if hook_mode.use_xsched():
    # if torch_col.is_enable_xsched():
    #     from torch_col import xsched
    #     xsched.register_stream(stream)
    #     print("CUDA Stream create with xsched registered.")
    # else:
    #     print("CUDA Stream create without xsched.")
    # with torch.cuda.stream(stream):
    #     train(train_mode, hook_mode, num_epoch, batch_size, global_batch_size=global_batch_size)


if __name__ == '__main__':
    main()