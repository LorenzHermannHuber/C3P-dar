"""Pixel-shuffle decoding head with local convolution and window attention."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class LocalConvResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
        )

    def forward(self, x):
        return F.gelu(x + self.block(x))


class WindowResidualAttentionBlock(nn.Module):
    """Local convolution, nonoverlapping (optionally offset) window attention and a residual MLP."""

    def __init__(self, channels, heads=4, window_size=8, shift_size=0):
        super().__init__()
        self.shift_size = shift_size
        self.local = LocalConvResidualBlock(channels)
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, channels * 2),
                                 nn.GELU(), nn.Linear(channels * 2, channels))
        self.window_size = window_size
        # CUDA SDPA kernels can limit the batch of windows to 65535.
        self.window_batch_size = 4096

    def forward(self, x):
        x = self.local(x)
        b, c, h, w = x.shape
        size = self.window_size
        shift = self.shift_size
        # Offset windows by masked padding instead of rolling: BEV edges never join.
        padding = (shift, -(w + shift) % size, shift, -(h + shift) % size)
        padded = F.pad(x, padding)
        hp, wp = padded.shape[-2:]
        tokens = padded.reshape(b, c, hp // size, size, wp // size, size)
        tokens = tokens.permute(0, 2, 4, 3, 5, 1).reshape(-1, size * size, c)
        valid = F.pad(x.new_ones((b, 1, h, w)), padding)
        mask = valid.reshape(b, 1, hp // size, size, wp // size, size)
        mask = mask.permute(0, 2, 4, 3, 5, 1).reshape(-1, size * size) == 0
        normalized = self.norm(tokens)
        attended = []
        for start in range(0, normalized.shape[0], self.window_batch_size):
            window = normalized[start:start + self.window_batch_size]
            attended.append(self.attn(window, window, window,
                                      key_padding_mask=mask[start:start + self.window_batch_size],
                                      need_weights=False)[0])
        tokens = tokens + torch.cat(attended, dim=0)
        tokens = tokens + self.mlp(tokens)
        return tokens.reshape(b, hp // size, wp // size, size, size, c).permute(
            0, 5, 1, 3, 2, 4).reshape(b, c, hp, wp)[..., shift:shift + h, shift:shift + w]


def _icnr_init(conv, upscale_factor):
    """Identical pixel-shuffle subkernels reduce checkerboard artifacts at init."""
    scale_area = upscale_factor ** 2
    subkernel = torch.empty_like(conv.weight[: conv.out_channels // scale_area])
    nn.init.kaiming_normal_(subkernel, mode="fan_in", nonlinearity="relu")
    with torch.no_grad():
        conv.weight.copy_(subkernel.repeat_interleave(scale_area, dim=0))
        if conv.bias is not None:
            conv.bias.zero_()


class AttentionPixelShuffleHead(nn.Module):
    """(B,D,h,w) patch features -> (B,3,16h*f,16w*f) fields: two 4x stages, plus a final
    stage with factor f = final_upsample_factor when output_scale is 2 (f = 1 otherwise)."""

    def __init__(self, dim, channels=(128, 64, 32), blocks=2, heads=4, window_size=8, output_scale=2,
                 grad_checkpointing=False, projection_kernel_size=1, shifted_windows=False,
                 final_upsample_factor=2):
        super().__init__()
        if output_scale not in (1, 2) or len(channels) != 3:
            raise ValueError("Attention head requires output_scale 1 or 2 and three widths")
        self.output_scale = output_scale
        self.grad_checkpointing = grad_checkpointing
        self.stages = nn.ModuleList()
        previous = dim
        for width, factor in zip(channels[:2] if output_scale == 1 else channels, (4, 4, final_upsample_factor)):
            projection = nn.Conv2d(previous, width * factor ** 2, projection_kernel_size,
                                   padding=projection_kernel_size // 2)
            _icnr_init(projection, factor)
            self.stages.append(nn.Sequential(projection, nn.PixelShuffle(factor), *[
                WindowResidualAttentionBlock(width, heads, window_size,
                                             shift_size=window_size // 2 if shifted_windows and index % 2 else 0)
                for index in range(blocks)]))
            previous = width
        self.readout = nn.Conv2d(previous, 3, 1)

    def forward(self, x):
        for stage in self.stages:
            for layer in stage:
                if self.grad_checkpointing and self.training and torch.is_grad_enabled():
                    x = checkpoint(layer, x, use_reentrant=False)
                else:
                    x = layer(x)
        return self.readout(x)
