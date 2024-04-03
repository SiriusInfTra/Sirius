import os
# os.environ['COLOCATE_HAS_SERVER'] = '1'
# os.environ['HAS_SHARED_TENSOR_SERVER'] = '0'
# os.environ['USE_SHARED_TENSOR'] = '0'
# os.environ['SHARED_TENSOR_POOL_GB'] = '13.5'
import random
from typing import Optional
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torchvision import models
from torch.utils.data import DataLoader
import argparse

import os
import torch_col
import numpy as np
from torch_col import MemoryPool, EventManager, TrainMode, HookMode
from torch_col import ColocateAdjustL1Exception, SwitchL1Exception
from torch_col import CustomeDynamicBatchDataset


# torch_col.disable_release_saved_tensor()
class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        # 输入图像大小为224x224x3
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 112 * 112, 10)

    def forward(self, x):
        x = x + 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 32 * 112 * 112)
        # if random.uniform(0, 1) > 0.5:

        #     raise ColocateAdjustL1Exception()
        x = self.fc1(x)
        return x

def train(train_mode: TrainMode, hook_mode: HookMode, num_epoch: int, batch_size: int, trace_output: Optional[list], trace_input: Optional[list]):
    if torch_col.use_shared_tensor():
        torch_col.tag_model_start()

    torch_col.train_model = model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT).cuda()

    # torch_col.train_model = model = SimpleConvNet().cuda()
    print(f"Train params memory usage: {torch_col.MemoryPool.get_memory_usage() * 1024:.2f}M")

    criterion = nn.CrossEntropyLoss().cuda(0)
    optimizer = torch.optim.SGD(model.parameters(), 0.1, 
                                momentum=0.9, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    print(f"Train after init memory pool usage: {MemoryPool.get_memory_usage() * 1024:.2f}M")
    
    hook = torch_col.get_hook(train_mode, hook_mode, num_epoch, batch_size)
    hook.register_pytorch_hook([model, criterion])

    # dummy data, todo: learning rate auto scaling
    train_dataset = CustomeDynamicBatchDataset(1000, (3, 224, 224), 10, batch_size, hook, trace_input)
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
            if trace_output is not None or trace_input is not None:
                rng_cuda = torch.cuda.get_rng_state()
                rng_cpu = torch.get_rng_state()
                rng_py = random.getstate()
                rng_np = np.random.get_state()
            try:
                tried_batch += 1
                total_tried_batch += 1
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    output = model(images)
                    loss = criterion(output, targets)
                if trace_input is not None or trace_output is not None:
                    print(f'batch_size = {len(images)} loss = fp{loss.item()}')
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
                if trace_input is not None:
                    maybe_trace_item = trace_input[train_dataset.trace_idx]
                    assert maybe_trace_item['epoch'] == epoch, f"{maybe_trace_item['epoch']} vs {epoch}, {epoch}:{i}"
                    # assert maybe_trace_item['micro_epoch'] == i, f"{maybe_trace_item['micro_epoch']} vs {i}, {epoch}:{i}"
                    assert maybe_trace_item['batch_size'] == len(images)
                    maybe_trace_item['loss_ground_truth'] = f'fp{loss.item()}'
                    maybe_trace_item['correct'] = 1 if maybe_trace_item['loss_ground_truth'] == maybe_trace_item['loss'] else 0
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
                if trace_output is not None or trace_input is not None:
                    torch.cuda.set_rng_state(rng_cuda)
                    torch.set_rng_state(rng_cpu)
                    random.setstate(rng_py)
                    np.random.set_state(rng_np)
            else:
                # torch.cuda.current_stream().synchronize()
                batch_event.tag = 'finish'
                train_dataset.next_batch()
                if trace_output is not None:
                    new_data = {'epoch': epoch, 'micro_epoch': i, 'batch_size': len(images), 'loss': f'fp{loss.item()}'} 
                    trace_output.append(new_data)
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
    parser.add_argument('--use-xsched', type=bool)
    args = parser.parse_args()
    
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    train_mode = [train_mode for train_mode in TrainMode if train_mode.value == args.train_mode][0]
    hook_mode = [hook_mode for hook_mode in HookMode if hook_mode.value == args.hook_mode][0]
    
    trace_output = [] if 'COL_TRACE_OUTPUT' in os.environ else None
    trace_input = pd.read_csv(os.environ['COL_TRACE_INPUT']).to_dict(orient='records') if 'COL_TRACE_INPUT' in os.environ else None
    print(f"ResNet152 training, batch-size={batch_size}, num-epoch={num_epoch}, train-mode={train_mode}, hook-mode={hook_mode}.")
    
    if trace_output is not None or trace_input is not None:  
        torch.backends.cudnn.benchmark = False   
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        print('Enable deterministic validation.')

    stream = torch.cuda.Stream()
    if hook_mode.use_xsched():
        from torch_col import xsched
        xsched.register_stream(stream)
        print("CUDA Stream create with xsched registered.")
    else:
        print("CUDA Stream create without xsched.")
    with torch.cuda.stream(stream):
        train(train_mode, hook_mode, num_epoch, batch_size, trace_output, trace_input)
    EventManager.dump(args.train_profile, train_mode)
    if trace_output is not None:
        pd.DataFrame(trace_output).to_csv(os.environ['COL_TRACE_OUTPUT'], index=None)
    if trace_input is not None:
        pd.DataFrame(trace_input).to_csv(os.environ['COL_TRACE_INPUT'], index=None)
if __name__ == '__main__':
    main()