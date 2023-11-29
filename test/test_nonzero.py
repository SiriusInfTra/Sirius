import torch

import torch_col


def print_storage(t: torch.Tensor):
    print(f"data_ptr={t.data_ptr()}, offset={t.storage_offset()}, num_e={t.numel()}.")

def print_tensor(t: torch.Tensor):
    print(f"tensor={t}")
    print(f"shape={t.shape}, stride{t.stride()}")


x = torch.ones(60).cuda()
y = torch.ones(60).cuda()
x_view = x.as_strided(size=[4, 3, 3], stride=[15, 5, 2])
y_view = y.as_strided(size=[4, 3, 3], stride=[15, 5, 2])
# print_tensor(x_view)
# print_tensor(y_view)
z = x_view + y_view
print_tensor(z)
