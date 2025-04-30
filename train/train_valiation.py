from __future__ import annotations

import os
import random
from typing import Optional
import numpy as np
import pandas as pd
from torch import nn
import torch
from torch.nn import functional as F
from torch_col import DynamicBatchDataset

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



def val_begin():
    global trace_input
    global trace_output
    trace_output = [] if '"COLTRAIN_TRACE_OUTPUT' in os.environ else None
    trace_input = pd.read_csv(os.environ['"COLTRAIN_TRACE_INPUT']).to_dict(orient='records') if '"COLTRAIN_TRACE_INPUT' in os.environ else None
    if trace_output is not None or trace_input is not None:  
        torch.backends.cudnn.benchmark = False   
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        print('Enable deterministic validation.')

def val_end():
    global trace_input
    global trace_output
    if trace_output is not None:
        pd.DataFrame(trace_output).to_csv(os.environ['"COLTRAIN_TRACE_OUTPUT'], index=None)
    if trace_input is not None:
        pd.DataFrame(trace_input).to_csv(os.environ['"COLTRAIN_TRACE_INPUT'], index=None)
        
def debug_print_loss(batch_size: int, loss: torch.Tensor):
    if trace_input is not None or trace_output is not None:
        print(f'batch_size = {batch_size} loss = fp{loss.item()}')

def make_rng_state_checkpoint():
    global trace_input
    global trace_output
    global rng_state
    if trace_output is not None or trace_input is not None:
        rng_cuda = torch.cuda.get_rng_state()
        rng_cpu = torch.get_rng_state()
        rng_py = random.getstate()
        rng_np = np.random.get_state()
        rng_state = rng_cuda, rng_cpu, rng_py, rng_np

def recover_rng_state():
    global trace_input
    global trace_output
    global rng_state
    if trace_output is not None or trace_input is not None:
        rng_cuda, rng_cpu, rng_py, rng_np = rng_state
        torch.cuda.set_rng_state(rng_cuda)
        torch.set_rng_state(rng_cpu)
        random.setstate(rng_py)
        np.random.set_state(rng_np)

def record_completed_batch(train_dataset: DynamicBatchDataset, epoch: int, batch: int, batch_size: int, loss: torch.Tensor):
    global trace_input
    global trace_output
    if trace_input is not None:
        maybe_trace_item = trace_input[train_dataset.trace_idx]
        assert maybe_trace_item['epoch'] == epoch, f"{maybe_trace_item['epoch']} vs {epoch}, {epoch}:{batch}"
        # assert maybe_trace_item['micro_epoch'] == i, f"{maybe_trace_item['micro_epoch']} vs {i}, {epoch}:{i}"
        assert maybe_trace_item['batch_size'] == batch_size
        maybe_trace_item['loss_ground_truth'] = f'fp{loss.item()}'
        maybe_trace_item['correct'] = 1 if maybe_trace_item['loss_ground_truth'] == maybe_trace_item['loss'] else 0
    if trace_output is not None:
        new_data = {'epoch': epoch, 'micro_epoch': batch, 'batch_size': batch_size, 'loss': f'fp{loss.item()}'} 
        trace_output.append(new_data)
        
def get_trace_input():
    global trace_input
    return trace_input