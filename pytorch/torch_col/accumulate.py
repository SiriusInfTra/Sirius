import torch
from collections import defaultdict
from typing import Optional


class GradAccumulator:
    def __init__(self, model):
        self.param_list = []
        self.param_grads = defaultdict(dict)

        for p in model.parameters():
            p: torch.Tensor
            if p.requires_grad:
                self.param_list.append(p)
                self.param_grads[p]['global'] = torch.zeros_like(p)
                self.param_grads[p]['local'] = None

    def accumulate(self):
        for p in self.param_list:
            self.param_grads[p]['global'] += p.grad 
            p.grad.zero_()

    def step(self, 
             optmizer: torch.optim.Optimizer, 
             scaler: Optional[torch.cuda.amp.GradScaler]=None, 
             do_zero_grad=True
    ):
        for p in self.param_list:
            assert p.grad is not None
            self.param_grads[p]['local'] = p.grad
            p.grad = self.param_grads[p]['global']
        if scaler is not None:
            scaler.step(optmizer)
            scaler.update()
        else:
            optmizer.step()
        for p in self.param_list:
            p.grad = self.param_grads[p]['local']

        if do_zero_grad:
            self.zero_grad()

    def zero_grad(self):
        for p in self.param_list:
            self.param_grads[p]['global'].zero_()

    
