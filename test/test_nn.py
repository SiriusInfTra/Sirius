import os
os.environ['USE_SHARED_TENSOR'] = "1"
os.environ['SHARED_TENSOR_POOL_GB'] = "12"
os.environ['SHARED_TENSOR_HAS_SERVER'] = "1"
os.environ['GLOG_logtostderr'] = "1"

import torch
import torch_col
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18

from io import BytesIO
from typing import List

# ignore dropout to maintain deterministic behavior
class Mnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def TestFwdFactory(model:nn.Module, input_shape:List):
    input_data_cpu = torch.randn(input_shape)
    input_data_cuda = input_data_cpu.cuda()


    model_cpu = model.eval()
    model_data_in_memory = BytesIO()
    torch.save(model_cpu, model_data_in_memory)
    output_data_cpu = model_cpu(input_data_cpu)
    
    def test_fwd_fn():
        model_data_in_memory.seek(0)
        model_cuda = torch.load(model_data_in_memory, map_location='cpu').cuda().eval()
        output_data_cuda = model_cuda(input_data_cuda)
        output_data_cuda_cpu = output_data_cuda.cpu()
        return torch.sum(torch.abs(output_data_cuda.cpu() - output_data_cpu))
    
    return test_fwd_fn

def TestBwdFactory(model:nn.Module, input_shape:List, criterion, target):
    input_data_cpu = torch.randn(input_shape, requires_grad=False)
    input_data_cuda = input_data_cpu.cuda()

    input_data_cpu.requires_grad = True
    input_data_cuda.requires_grad = True

    criterion_cuda = criterion.cuda()
    target_cuda = target.cuda()

    model_cpu = model
    model_data_in_memory = BytesIO()
    torch.save(model_cpu, model_data_in_memory)
    output_data_cpu = model_cpu(input_data_cpu)
    loss_cpu = criterion(output_data_cpu, target)
    loss_cpu.backward()
    grad_cpu = input_data_cpu.grad

    def test_bwd_fn():
        input_data_cuda.grad = None
        model_data_in_memory.seek(0)
        model_cuda = torch.load(model_data_in_memory, map_location='cpu').cuda().eval()
        output_data_cuda = model_cuda(input_data_cuda)
        loss_cuda = criterion_cuda(output_data_cuda, target_cuda)
        loss_cuda.backward()
        grad_cuda = input_data_cuda.grad
        return torch.max(torch.abs(grad_cuda.cpu() - grad_cpu))

    return test_bwd_fn


test_mnist_fwd_fn = TestFwdFactory(model=Mnist(), input_shape=(1, 1, 28, 28))
print(test_mnist_fwd_fn())

test_resnet18_fwd_fn = TestFwdFactory(model=resnet18(), input_shape=(1, 3, 224, 224))
print(test_resnet18_fwd_fn())

test_mnist_bwd_fn = TestBwdFactory(model=Mnist(), input_shape=(1, 1, 28, 28), 
    criterion=nn.NLLLoss(), target=torch.tensor([1]))
print(test_mnist_bwd_fn())

test_resnet18_bwd_fn = TestBwdFactory(model=resnet18(), input_shape=(32, 3, 224, 224),
    criterion=nn.NLLLoss(), target=torch.tensor(list(range(32))))
print(test_resnet18_bwd_fn())

for i in range(100):
    assert test_mnist_fwd_fn() < 1e-3
    assert test_resnet18_fwd_fn() < 1e-2

    assert test_mnist_bwd_fn() < 1e-2
    assert test_resnet18_bwd_fn() < 1e-2
