import torch
from torch import nn
import torch_col.xsched
from torchvision import models
from torch.utils.data import DataLoader
import argparse
import sys

import torch_col
torch_col.torch_col_init()

from torch_col import MemoryPool, EventManager, TrainMode, ColocateCtrlHookMode
from torch_col import ColocateAdjustL1Exception, SwitchL1Exception
from torch_col import DynamicBatchDataset
import train_valiation
from typing import Optional


def setup():
    train_valiation.val_begin()
    stream = torch.cuda.Stream()
    if torch_col.is_enable_xsched():
        torch_col.xsched.register_stream(stream)
        print("CUDA Stream create with xsched registered.")
    else:
        print("CUDA Stream create without xsched.")
    torch.cuda.set_stream(stream)

def cleanup():
    if torch_col.is_enable_xsched():
        torch_col.xsched.unregister_all_streams()
    train_valiation.val_end()


def train(train_mode: TrainMode, hook_mode: ColocateCtrlHookMode, 
          num_epoch: int, batch_size: int, global_batch_size: Optional[int] = None):
    if torch_col.is_enable_shared_tensor():
        torch_col.tag_model_start()

    torch_col.train_model = model = models.swin_b(weights=models.Swin_B_Weights.DEFAULT).cuda()

    print(f"Train params memory usage: {torch_col.MemoryPool.get_memory_usage() * 1024:.2f}M")

    criterion = nn.CrossEntropyLoss().cuda(0)
    optimizer = torch.optim.SGD(model.parameters(), 0.1, 
                                momentum=0.9, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    print(f"Train after init memory pool usage: {MemoryPool.get_memory_usage() * 1024:.2f}M")
    
    hook = torch_col.create_colocate_ctrl(train_mode, hook_mode, num_epoch, batch_size)
    torch_col.init_train_info(batch_size, batch_size)
    hook.register_pytorch_hook([model, criterion])
    checkpoint_micro_batch = hook.train_mode.is_kill_batch()

    # dummy data
    train_dataset = DynamicBatchDataset(
        model_name='swin_b', size=1000, max_batch_size=batch_size, 
        hook=hook, trace=train_valiation.get_trace_input(),
        input_shape=(3, 224, 224), num_class=10, 
        max_global_batch_size=global_batch_size,
        checkpoint_micro_batch=checkpoint_micro_batch)
    train_loader = DataLoader(train_dataset, batch_size=None, 
                              shuffle=False, pin_memory=True, drop_last=False, num_workers=0)

    total_killed_batch = 0
    total_finished_batch = 0
    total_tried_batch = 0

    model.train()
    hook.train_start()

    torch_col.util.initialize_sgd_optimizer(model, optimizer)
    if train_dataset.checkpoint_micro_batch:
        grad_accumulator = torch_col.GradAccumulator(model)
    else:
        grad_accumulator = None

    print('swin after initialize, allocated {} cached {}'.format(
        torch_col.MemoryPool.get_allocated_memory(), 
        torch_col.MemoryPool.get_memory_usage()),
        flush=True, file=sys.stderr)

    # print_opt(optimizer)

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
            # print(f'[epoch {epoch} batch {i}] batch size {len(images)}', flush=True, file=sys.stderr)
            images = batch['image']
            targets = batch['label']
            train_dataset.record_batch_event(epoch, i, len(images), train_dataset.global_batch_size)
            images: torch.Tensor = images.to('cuda:0', non_blocking=True)
            targets: torch.Tensor = targets.to('cuda:0', non_blocking=True)
            train_valiation.make_rng_state_checkpoint()
            try:
                tried_batch += 1
                total_tried_batch += 1
                optimizer.zero_grad(set_to_none=False)
                with torch.cuda.amp.autocast(cache_enabled=False):
                    output = model(images)
                    loss = criterion(output, targets)
                    train_dataset.scale_loss(loss)
                train_valiation.debug_print_loss(len(images), loss)
                scaler.scale(loss).backward()
                train_dataset.step_stage(hook, optimizer, scaler=scaler, grad_accumulator=grad_accumulator)
                # if train_dataset.is_do_checkpoint_micro_batch():
                #     with hook.steps_no_interrupt():
                #         grad_accumulator.accumulate()
                #         if train_dataset.is_do_step():
                #             grad_accumulator.step(optimizer, scaler)
                # else:
                #     if train_dataset.is_do_step():
                #         with hook.steps_no_interrupt():
                #             step_event = EventManager.record_event('optimizer_step')
                #             scaler.step(optimizer)
                #             scaler.update()
                #         EventManager.record_event('', step_event)
                finished_batch += 1
                total_finished_batch += 1
                batch_cnt += 1
                finished_imgs += len(images)

            except (ColocateAdjustL1Exception, SwitchL1Exception, torch_col.EngineColocateAdjustL1Exception) as e:
                if epoch == 0 and i == 0:
                    raise RuntimeError("micro batch 0 could not be interrupted.")
                # for fast develop, should merge to hook.py
                if isinstance(e, torch_col.EngineColocateAdjustL1Exception):
                    torch_col.xsched.kill_batch()
                killed_batch += 1
                total_killed_batch += 1
                train_dataset.cancel_micro_batch()
                with EventManager.record_duration_event(f'batch_exception_{epoch:02d}_{i:03d}_{len(images):02d}'):
                    # cuda has alreadly synced
                    hook.release_and_reply()
                    print(f'[{e}] batch_size: {len(images)} -> {train_dataset.batch_size}.')
                train_valiation.recover_rng_state()
            else:
                # torch.cuda.current_stream().synchronize()
                train_valiation.record_completed_batch(train_dataset, epoch, i, len(images), loss)
                train_dataset.next_batch()
                if epoch == 0 and i == 0:
                    if torch_col.is_enable_shared_tensor():
                        torch_col.tag_model_end()
            if hook_mode.use_xsched():
                torch_col.xsched.initial_kill_batch(epoch, i)
        EventManager.record_event('', epoch_event)

        mem_info = f'mem {MemoryPool.get_memory_usage():.2f}Gb'
        batch_info = f'batch cnt {batch_cnt} avg {epoch_event.duration/batch_cnt:.1f}ms'
        if train_mode.is_kill_batch():
            batch_info += f' | try {tried_batch} kill {killed_batch}, {killed_time*1e3:.1f}ms finish {finished_batch}, {finished_time*1e3:.1f}ms'
        if global_batch_size is not None:
            batch_info += f' | num_rollback_sampels {train_dataset.num_rollback_samples_in_epoch}'
        print('[{} epoch {}] {:.3f}s | {} | batch-size {} | micro-batch-size {} | {} | thpt {:.2f} | wait_bs_valid {:.3f}s | loss {:.6f}'.format(
                model.__class__.__name__, epoch, epoch_event.duration / 1e3,
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
    hook_mode = torch_col.get_colocate_ctrl_hook_mode()
    

    print(f"Swin Transformer training, batch-size={batch_size}, num-epoch={num_epoch}, train-mode={train_mode}, hook-mode={hook_mode}.")
    
    setup()
    train(train_mode, hook_mode, num_epoch, batch_size, global_batch_size=global_batch_size)
    cleanup()
    EventManager.dump(args.train_profile, train_mode)

if __name__ == '__main__':
    main()