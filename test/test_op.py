
# %%
%load_ext dotenv
%dotenv
import os
print('CUDA_LAUNCH_BLOCKING', os.environ['CUDA_LAUNCH_BLOCKING'])

import torch
import torch_col
a = torch.empty((2, 2), device='cuda:0')

# b = torch.ones(4, device='cuda:0')

c = a.reshape(-1)



# %%
b = a.abs()
a[0] = 1
print(a)


# %%
b = torch.ones(5, device='cuda:0')
print(b)

# %%
print(a * b)
print(a + b)


# %%
