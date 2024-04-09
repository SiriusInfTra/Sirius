import torch
from torch import nn
from torchvision import models
from torch.utils.data import DataLoader
import argparse
import sys

import torch_col
from torch_col import MemoryPool, EventManager, TrainMode, HookMode
from torch_col import ColocateAdjustL1Exception, SwitchL1Exception
from torch_col import CustomeDynamicBatchDataset
import train_valiation

checkpoint_micro_batch = False

def train(train_mode: TrainMode, hook_mode: HookMode, num_epoch: int, batch_size: int):
    if torch_col.use_shared_tensor():
        torch_col.tag_model_start()

    torch_col.train_model = model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT).cuda()

    print(f"Train params memory usage: {torch_col.MemoryPool.get_memory_usage() * 1024:.2f}M")

    criterion = nn.CrossEntropyLoss().cuda(0)
    optimizer = torch.optim.SGD(model.parameters(), 0.1, 
                                momentum=0.9, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    print(f"Train after init memory pool usage: {MemoryPool.get_memory_usage() * 1024:.2f}M")
    
    hook = torch_col.get_hook(train_mode, hook_mode, num_epoch, batch_size)
    hook.register_pytorch_hook([model, criterion])

    # dummy data
    train_dataset = CustomeDynamicBatchDataset(1000, (3, 224, 224), 10, batch_size, hook, 
                                               train_valiation.get_trace_input())
    train_loader = DataLoader(train_dataset, batch_size=None, 
                              shuffle=False, pin_memory=True, drop_last=False, num_workers=0)

    total_killed_batch = 0
    total_finished_batch = 0
    total_tried_batch = 0

    model.train()
    hook.train_start()

    for epoch in range(num_epoch):
        epoch_event = EventManager.record_event(f'epoch_{epoch:02d}')
        batch_cnt = 0
        killed_batch = 0
        finished_batch = 0
        tried_batch = 0
        killed_time = 0
        finished_time = 0
        wait_bs_valid_sec = 0 # add infer may cause batch size <= 0
        finished_imgs = 0
        for i, (images, targets) in enumerate(train_loader):
            batch_event = EventManager.record_event(f'batch_{epoch:02d}_{i:03d}_{len(images):02d}')
            images: torch.Tensor = images.to('cuda:0', non_blocking=True)
            targets: torch.Tensor = targets.to('cuda:0', non_blocking=True)
            train_valiation.make_rng_state_checkpoint()
            try:
                tried_batch += 1
                total_tried_batch += 1
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    output = model(images)
                    loss = criterion(output, targets)
                train_valiation.debug_print_loss(len(images), loss)
                scaler.scale(loss).backward()
                with hook.steps_no_interrupt():
                    event = EventManager.record_event('optimizer_step')
                    scaler.step(optimizer)
                    scaler.update()
                event = EventManager.record_event('', event)
                # finished_time += time.time() - micro_batch_begin
                finished_batch += 1
                total_finished_batch += 1
                batch_cnt += 1
                finished_imgs += len(images)

            except (ColocateAdjustL1Exception, SwitchL1Exception, torch_col.EngineColocateAdjustL1Exception) as e:
                if epoch == 0 and i == 0:
                    raise RuntimeError("micro batch 0 could not be interrupted.")
                # for fast develop, should merge to hook.py
                if isinstance(e, torch_col.EngineColocateAdjustL1Exception):
                    xsched.kill_batch()
                killed_batch += 1
                total_killed_batch += 1
                batch_event.tag = 'cancel'
                with EventManager.record_duration_event(f'batch_exception_{epoch:02d}_{i:03d}_{len(images):02d}'):
                    # cuda has alreadly synced
                    hook.release_and_reply()
                    print(f'[{e}] batch_size: {len(images)} -> {train_dataset.batch_size}.')
                if not checkpoint_micro_batch:
                    train_dataset.rollback_micro_batch()
                train_valiation.recover_rng_state()
            else:
                # torch.cuda.current_stream().synchronize()
                batch_event.tag = 'finish'
                train_valiation.record_completed_batch(train_dataset, epoch, i, len(images), loss)
                train_dataset.next_batch()
                if epoch == 0 and i == 0:
                    if torch_col.use_shared_tensor():
                        torch_col.tag_model_end()
            EventManager.record_event('', batch_event)
            if hook_mode.use_xsched():
                from torch_col import xsched
                xsched.initial_kill_batch(epoch, i)
        EventManager.record_event('', epoch_event)

        mem_info = f'mem {MemoryPool.get_memory_usage():.2f}Gb'
        batch_info = f'batch cnt {batch_cnt} avg {epoch_event.duration/batch_cnt:.1f}ms'
        if train_mode.is_kill_batch():
            batch_info += f' | try {tried_batch} kill {killed_batch}, {killed_time*1e3:.1f}ms finish {finished_batch}, {finished_time*1e3:.1f}ms'
        print('[{} epoch {}] {:.3f}s | {} | batch-size {} | micro-batch-size {} | {} | thpt {:.2f} | wait_bs_valid {:.3f}s | loss {:.6f}'.format(
                model.__class__.__name__, epoch, epoch_event.duration / 1e3,
                batch_info, batch_size, train_dataset.batch_size,
                mem_info, finished_imgs / (epoch_event.duration / 1e3), wait_bs_valid_sec, loss.item()), flush=True)
    

    if train_mode.is_kill_batch():
        print("[{}] Epoch x Batch {} | Batch Total Tried {} Killed {} Finished {}".format(
            model.__class__.__name__,
            num_epoch * batch_size, total_tried_batch, total_killed_batch, total_finished_batch))
    
    hook.train_end()
    hook.stop()

def main():
    parser = argparse.ArgumentParser('Train Resnet')    
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-epoch', type=int, default=15)
    parser.add_argument('--train-mode', type=str, default=TrainMode.COLOCATE_L1.value, choices=[train_mode.value for train_mode in TrainMode])
    parser.add_argument('--train-profile', type=str, default='train-profile.csv')
    parser.add_argument('--hook-mode', default=HookMode.XSCHED_SYNC.value, choices=[hook_mode.value for hook_mode in HookMode])
    parser.add_argument('--use-xsched', type=bool) # not used args
    args = parser.parse_args()
    
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    train_mode = [train_mode for train_mode in TrainMode if train_mode.value == args.train_mode][0]
    hook_mode = [hook_mode for hook_mode in HookMode if hook_mode.value == args.hook_mode][0]
    

    print(f"ResNet152 training, batch-size={batch_size}, num-epoch={num_epoch}, train-mode={train_mode}, hook-mode={hook_mode}.")
    train_valiation.val_begin()
    stream = torch.cuda.Stream()
    if hook_mode.use_xsched():
        from torch_col import xsched
        xsched.register_stream(stream)
        print("CUDA Stream create with xsched registered.")
    else:
        print("CUDA Stream create without xsched.")
    with torch.cuda.stream(stream):
        train(train_mode, hook_mode, num_epoch, batch_size)
    train_valiation.val_end()
    EventManager.dump(args.train_profile, train_mode)

if __name__ == '__main__':
    main()