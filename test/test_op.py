
# %%
# %env CUDA_VISIBLE_DEVICES="GPU-ea8006f2-470f-f147-2425-74cede8f6cd8"
# %env USE_SHARED_TENSOR=1
# %env HAS_SHARED_TENSOR_SERVER=0
# %env COL_SHARED_TENSOR_POOL_GB=12

import torch
import torch_col

# %%

a = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda')
b = a.as_strided([3, 2], [1, 3])
# a = torch.empty((2, 2), device='cuda:0')

# # b = torch.ones(4, device='cuda:0')
print(hex(b.data_ptr()))
c = b.select(0, 0)
print(b)

# c = a.reshape(-1)



# %%
# b = a.abs()
# a[0] = 1
# print(a)


# # %%
# b = torch.ones(5, device='cuda:0')
# print(b)

# # %%
# print(a * b)
# print(a + b)


# %%
