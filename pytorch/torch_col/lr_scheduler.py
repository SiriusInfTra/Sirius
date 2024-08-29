
from timm.scheduler.cosine_lr import CosineLRScheduler
import torch
from torch.optim import Optimizer
import torch_col
    
def get_imagenet_optimizer_and_scheduler(model: torch.nn.Module, global_batch_size: int, num_steps: int, warmup_steps):
    # Swin-Transformer 
    # configs/swin/swin_base_patch4_window7_224.yaml
    # config.py
    WEIGHT_DECAY = 0.05
    BASE_LR = 5e-4
    WARMUP_LR = 5e-7
    MIN_LR = 5e-6
    WARMUP_PREFIX = True

    
    world_size = torch_col.get_train_world_size()

    linear_scaled_lr = BASE_LR * global_batch_size * world_size / 512.0
    linear_scaled_warmup_lr = WARMUP_LR * global_batch_size * world_size / 512.0
    linear_scaled_min_lr = MIN_LR * global_batch_size * world_size / 512.0
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        linear_scaled_lr,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY
    )
    
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=(num_steps - warmup_steps) if WARMUP_PREFIX else num_steps,
        lr_min=linear_scaled_min_lr,
        warmup_lr_init=linear_scaled_warmup_lr,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
        warmup_prefix=WARMUP_PREFIX,
    )
    

    return optimizer, scheduler 