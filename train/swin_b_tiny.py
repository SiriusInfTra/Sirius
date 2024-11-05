from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Helper Functions
def window_partition(x: torch.Tensor, window_size: int):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int, C: int):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
    return x


# Model Components
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, dropout_rate=0.0):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)

        num_window_elements = (2 * window_size - 1) * (2 * window_size - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(num_window_elements, num_heads))
        
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_patch, num_heads, window_size=7, shift_size=0, num_mlp=1024, qkv_bias=True, dropout_rate=0.0):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.num_patch = num_patch
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_mlp = num_mlp

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias, dropout_rate=dropout_rate)
        self.drop_path = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, num_mlp),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_mlp, dim),
            nn.Dropout(dropout_rate),
        )

        if min(self.num_patch) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.num_patch)

    def forward(self, x: torch.Tensor):
        H, W = self.num_patch
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W, C)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

# Model Definition
class SwinTransformer(nn.Module):
    def __init__(
        self, 
        input_shape: tuple[int, int], 
        patch_size: tuple[int, int], 
        qkv_bias: bool, 
        num_classes: int, 
        embed_dim: int, 
        num_heads: int, 
        num_mlp: int, 
        window_size: int, 
        shift_size: int, 
        dropout_rate: float
    ):
        super(SwinTransformer, self).__init__()
        num_patch_x = input_shape[0] // patch_size[0]
        num_patch_y = input_shape[1] // patch_size[1]
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = nn.Linear(3 * patch_size[0] * patch_size[1], embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, num_patch_x * num_patch_y, embed_dim))
        
        self.swin_block1 = SwinTransformerBlock(embed_dim, (num_patch_x, num_patch_y), num_heads, window_size, 0, num_mlp, qkv_bias, dropout_rate)
        self.swin_block2 = SwinTransformerBlock(embed_dim, (num_patch_x, num_patch_y), num_heads, window_size, shift_size, num_mlp, qkv_bias, dropout_rate)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        x = self.patch_embed(x) + self.positional_encoding
        x = self.swin_block1(x)
        x = self.swin_block2(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x