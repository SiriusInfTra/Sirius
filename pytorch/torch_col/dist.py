import torch_col

def wait_barrier():
    torch_col.DistTrainSync.WaitBarrier()