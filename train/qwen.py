import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import argparse
import os, sys

import torch_col
from torch_col import MemoryPool, EventManager
from torch_col import dynamic_batch
from typing import Optional

def setup():
    import time
    torch_col.info(f'[Train PID {os.getpid()}] setup')
    torch_col.setup_colocate_training(0, 1, False, False)


def cleanup():
    torch_col.cleanup_colocate_training(False)


def train(num_epoch: int, batch_size: int,
          global_batch_size: Optional[int] = None):
    setup()
    torch_col.init_train_info(batch_size, batch_size,
                              model_name='qwen')
    
    train_mode = torch_col.get_colocate_train_mode()
    hook_mode = torch_col.get_colocate_ctrl_hook_mode()

    if torch_col.is_enable_shared_tensor():
        torch_col.tag_model_start()

    # TODO: flash attention
    config = AutoConfig.from_pretrained('Qwen/Qwen2-0.5B')
    # config._attn_implementation = 'sdpa'
    model: nn.Module = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-0.5B', config=config)
    model = model.cuda()
    print(f"Train params memory usage: {torch_col.MemoryPool.get_memory_usage() * 1024:.2f}M")

    optimizer = torch.optim.SGD(model.parameters(), 0.001,
                                momentum=0.9, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    print(f"Train after init memory pool usage: {MemoryPool.get_memory_usage() * 1024:.2f}M")
    
    col_ctrl = torch_col.create_colocate_ctrl(train_mode, hook_mode, num_epoch, batch_size)
    col_ctrl.register_pytorch_hook([model])

    checkpoint_micro_batch = col_ctrl.train_mode.is_kill_batch()
    enable_grad_accumulator = (global_batch_size is not None
                                or col_ctrl.train_mode.is_colocate())
    
    train_dataset, batch_manager = torch_col.init_dynamic_batch(
        dataset_size=1000,
        dataset_type=dynamic_batch.DatasetType.TEXT_GEN,
        dataset_config=dynamic_batch.TextDatasetConfig(128),
        batch_size=batch_size,
        global_batch_size=global_batch_size,
        enable_grad_accumulate=enable_grad_accumulator,
        checkpoint_micro_batch=checkpoint_micro_batch,
        lazy_batch_distributing=False,
        batch_distribute_policy=dynamic_batch.BatchDistributePolicy.BY_PERFORMANCE
    )

    model.train()
    col_ctrl.train_start()

    torch_col.util.initialize_sgd_optimizer(model, optimizer)
    if batch_manager._checkpoint_micro_batch:
        grad_accumulator = torch_col.GradAccumulator(model)
    else:
        grad_accumulator = None

    print('qwen-0.5B after initialize, allocated {} cached {}'.format(
        torch_col.MemoryPool.get_allocated_memory(), 
        torch_col.MemoryPool.get_memory_usage()),
        flush=True, file=sys.stderr)

    def iter_train_fn(batch: dynamic_batch.Batch):
        input_ids = batch['input_ids']
        with torch.cuda.amp.autocast(cache_enabled=False, dtype=torch.float16):
            output = model(**batch)
            loss = output.loss
            running_loss = loss.item() * len(input_ids)
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
            f'avg {epoch_duration/epoch_stat.finished_batch:.1f}ms')
        if train_mode.is_kill_batch():
            batch_info += (
                f' | try {epoch_stat.tried_batch} kill {epoch_stat.killed_batch}, '
                f'{epoch_stat.killed_time*1e3:.1f}ms finish {epoch_stat.finished_batch}, '
                f'{epoch_stat.finished_time*1e3:.1f}ms')
        if global_batch_size is not None:
            batch_info += f' | num_rollback_sampels {epoch_stat.num_rollback_batch}'

        print(f'[{model.__class__.__name__} epoch {epoch}] '
              f'{epoch_duration / 1e3:.3f}s | {batch_info} | batch-size {batch_size} '
              f'| {mem_info} | thpt {epoch_stat.finished_sample / (epoch_duration / 1e3):.3f} '
              f'| loss {running_loss:.6f}', 
              flush=True)

        if col_ctrl.can_exit_after_infer_worklaod_done():
            print('[hook] inference workload done, will exit training', flush=True)
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

    print(f"Qwen/Qwen2-0.5B training, batch-size={batch_size}, "
          f"num-epoch={num_epoch}")

    train(num_epoch, batch_size, 
          global_batch_size=global_batch_size)


if __name__ == '__main__':
    main()