"""DC-AE f32c32 (SANA 1.1) encoder and decoder, ported from MIT Han Lab's EfficientViT.

Key names match the pretrained safetensors (``encoder.*``, ``decoder.*``), so the
checkpoint loads without remapping. Only the ops used by this configuration are kept:
ResBlock stages 0-2, EfficientViT (LiteMLA scale 5 + GLUMBConv) stages 3-5, stride-2
convolution downsampling and nearest-interpolation upsampling with fixed shortcuts.
"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

WIDTHS = (128, 256, 512, 512, 1024, 1024)
ENCODER_DEPTHS = (2, 2, 2, 3, 3, 3)
DECODER_DEPTHS = (3, 3, 3, 3, 3, 3)
VIT_STAGES = (3, 4, 5)
SCALING_FACTOR = 0.41407


class RMSNorm2d(nn.Module):
    """Channel RMS norm; the reduction runs in FP32, the affine transform in the input dtype."""

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x * torch.rsqrt(x.float().square().mean(dim=1, keepdim=True) + self.eps).to(x.dtype)
        return x * self.weight.to(x.dtype).view(1, -1, 1, 1) + self.bias.to(x.dtype).view(1, -1, 1, 1)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=True, norm=False,
                 act=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, groups=groups,
                              bias=bias)
        self.norm = RMSNorm2d(out_channels) if norm else None
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, main, shortcut):
        super().__init__()
        self.main = main
        self.shortcut = shortcut

    def forward(self, x):
        return self.main(x) + self.shortcut(x)


class OpSequential(nn.Module):
    def __init__(self, ops):
        super().__init__()
        self.op_list = nn.ModuleList(ops)

    def forward(self, x):
        for op in self.op_list:
            x = op(x)
        return x


class Identity(nn.Module):
    def forward(self, x):
        return x


class ChannelAveraging(nn.Module):
    """Parameter-free downsampling shortcut: pixel-unshuffle, then average channel groups."""

    def __init__(self, in_channels, out_channels, factor):
        super().__init__()
        self.out_channels = out_channels
        self.factor = factor
        self.group_size = in_channels * factor ** 2 // out_channels

    def forward(self, x):
        x = F.pixel_unshuffle(x, self.factor)
        b, _, h, w = x.shape
        return x.view(b, self.out_channels, self.group_size, h, w).mean(dim=2)


class ChannelDuplicating(nn.Module):
    """Parameter-free upsampling shortcut: repeat channels, then pixel-shuffle."""

    def __init__(self, in_channels, out_channels, factor):
        super().__init__()
        self.factor = factor
        self.repeats = out_channels * factor ** 2 // in_channels

    def forward(self, x):
        return F.pixel_shuffle(x.repeat_interleave(self.repeats, dim=1), self.factor)


class InterpolateConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvLayer(in_channels, out_channels)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvLayer(channels, channels, bias=True, act=nn.SiLU())
        self.conv2 = ConvLayer(channels, channels, bias=False, norm=True)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class LiteMLA(nn.Module):
    """ReLU linear attention over 32-channel heads with one 5x5 multi-scale aggregation."""

    def __init__(self, channels, dim=32, eps=1e-15):
        super().__init__()
        self.dim = dim
        self.eps = eps
        heads = channels // dim
        self.qkv = ConvLayer(channels, 3 * channels, 1, bias=False)
        self.aggreg = nn.ModuleList([nn.Sequential(
            nn.Conv2d(3 * channels, 3 * channels, 5, padding=2, groups=3 * channels, bias=False),
            nn.Conv2d(3 * channels, 3 * channels, 1, groups=3 * heads, bias=False),
        )])
        self.proj = ConvLayer(2 * channels, channels, 1, bias=False, norm=True)

    @torch.autocast(device_type="cuda", enabled=False)
    def relu_linear_att(self, qkv):
        b, _, h, w = qkv.shape
        if qkv.dtype == torch.float16:
            qkv = qkv.float()
        qkv = qkv.reshape(b, -1, 3 * self.dim, h * w)
        q, k, v = qkv[:, :, :self.dim], qkv[:, :, self.dim:2 * self.dim], qkv[:, :, 2 * self.dim:]
        q, k = F.relu(q), F.relu(k)
        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=1)
        out = torch.matmul(torch.matmul(v, k.transpose(-1, -2)), q)
        if out.dtype == torch.bfloat16:
            out = out.float()
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)
        return out.reshape(b, -1, h, w)

    @torch.autocast(device_type="cuda", enabled=False)
    def relu_quadratic_att(self, qkv):
        b, _, h, w = qkv.shape
        qkv = qkv.reshape(b, -1, 3 * self.dim, h * w)
        q, k, v = qkv[:, :, :self.dim], qkv[:, :, self.dim:2 * self.dim], qkv[:, :, 2 * self.dim:]
        q, k = F.relu(q), F.relu(k)
        att_map = torch.matmul(k.transpose(-1, -2), q)
        dtype = att_map.dtype
        if dtype in (torch.float16, torch.bfloat16):
            att_map = att_map.float()
        att_map = (att_map / (att_map.sum(dim=2, keepdim=True) + self.eps)).to(dtype)
        return torch.matmul(v, att_map).reshape(b, -1, h, w)

    def forward(self, x):
        qkv = self.qkv(x)
        qkv = torch.cat([qkv] + [op(qkv) for op in self.aggreg], dim=1)
        h, w = qkv.shape[-2:]
        if h * w > self.dim:
            out = self.relu_linear_att(qkv).to(qkv.dtype)
        else:
            out = self.relu_quadratic_att(qkv)
        return self.proj(out)


class GLUMBConv(nn.Module):
    def __init__(self, channels, expand_ratio=4):
        super().__init__()
        mid = channels * expand_ratio
        self.inverted_conv = ConvLayer(channels, 2 * mid, 1, bias=True, act=nn.SiLU())
        self.depth_conv = ConvLayer(2 * mid, 2 * mid, 3, groups=2 * mid, bias=True)
        self.point_conv = ConvLayer(mid, channels, 1, bias=False, norm=True)
        self.glu_act = nn.SiLU()

    def forward(self, x):
        x, gate = self.depth_conv(self.inverted_conv(x)).chunk(2, dim=1)
        return self.point_conv(x * self.glu_act(gate))


class EfficientViTBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.context_module = ResidualBlock(LiteMLA(channels), Identity())
        self.local_module = ResidualBlock(GLUMBConv(channels), Identity())

    def forward(self, x):
        return self.local_module(self.context_module(x))


def _block(stage, width):
    return EfficientViTBlock(width) if stage in VIT_STAGES else ResidualBlock(ResBlock(width), Identity())


class Encoder(nn.Module):
    """(B,3,H,W) in [-1,1] -> (B,32,H/32,W/32) unscaled latents."""

    def __init__(self, in_channels=3, latent_channels=32):
        super().__init__()
        self.project_in = ConvLayer(in_channels, WIDTHS[0])
        stages = []
        for stage, (width, depth) in enumerate(zip(WIDTHS, ENCODER_DEPTHS)):
            ops = [_block(stage, width) for _ in range(depth)]
            if stage < len(WIDTHS) - 1:
                ops.append(ResidualBlock(ConvLayer(width, WIDTHS[stage + 1], stride=2),
                                         ChannelAveraging(width, WIDTHS[stage + 1], 2)))
            stages.append(OpSequential(ops))
        self.stages = nn.ModuleList(stages)
        self.project_out = ResidualBlock(OpSequential([ConvLayer(WIDTHS[-1], latent_channels)]),
                                         ChannelAveraging(WIDTHS[-1], latent_channels, 1))

    def forward(self, x):
        x = self.project_in(x)
        for stage in self.stages:
            x = stage(x)
        return self.project_out(x)


class Decoder(nn.Module):
    """(B,in_channels,h,w) -> (B,out_channels,32h,32w).

    ``in_channels`` other than the latent width uses a plain convolution input
    (no duplicating shortcut), as for decoding DiT features directly.
    """

    def __init__(self, in_channels=32, out_channels=3, latent_channels=32, gradient_checkpointing=False):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        if in_channels == latent_channels:
            self.project_in = ResidualBlock(ConvLayer(in_channels, WIDTHS[-1]),
                                            ChannelDuplicating(in_channels, WIDTHS[-1], 1))
        else:
            self.project_in = ConvLayer(in_channels, WIDTHS[-1])
        stages = []
        for stage, (width, depth) in enumerate(zip(WIDTHS, DECODER_DEPTHS)):
            ops = []
            if stage < len(WIDTHS) - 1:
                ops.append(ResidualBlock(InterpolateConv(WIDTHS[stage + 1], width),
                                         ChannelDuplicating(WIDTHS[stage + 1], width, 2)))
            ops.extend(_block(stage, width) for _ in range(depth))
            stages.append(OpSequential(ops))
        self.stages = nn.ModuleList(stages)
        self.project_out = OpSequential([RMSNorm2d(WIDTHS[0]), nn.ReLU(), ConvLayer(WIDTHS[0], out_channels)])

    def forward(self, x):
        def run(module, value):
            if self.gradient_checkpointing and torch.is_grad_enabled():
                return checkpoint(module, value, use_reentrant=False)
            return module(value)

        x = run(self.project_in, x)
        for stage in reversed(self.stages):
            x = run(stage, x)
        return run(self.project_out, x)


def load_pretrained(module, path, prefix):
    """Load ``prefix``-keys of the DC-AE safetensors into an Encoder/Decoder.

    Output convolutions with fewer channels are initialized with the mean over the RGB
    filters, other shape mismatches (the input projection of feature decoders) keep
    their fresh initialization. Returns the names of the keys that were not loaded.
    """
    from safetensors.torch import load_file

    path = Path(path)
    state = load_file(str(path / "model.safetensors" if path.is_dir() else path))
    state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    own = module.state_dict()
    skipped = []
    replaced = [key for key in state if key.startswith("project_in.main.") and key not in own]
    if replaced:
        # Feature decoders have a plain input convolution in place of DC-AE's residual block.
        for key in replaced:
            del state[key]
        skipped += [key for key in own if key.startswith("project_in.")]
    for key, value in list(state.items()):
        if key not in own:
            raise KeyError(f"Unexpected DC-AE key {prefix}{key}")
        if value.shape == own[key].shape:
            continue
        if key.startswith("project_out.op_list.2.conv."):
            state[key] = value.mean(dim=0, keepdim=True).expand_as(own[key]).clone()
        else:
            del state[key]
            skipped.append(key)
    missing, unexpected = module.load_state_dict(state, strict=False)
    if unexpected or set(missing) - set(skipped):
        raise KeyError(f"DC-AE mismatch: missing {missing}, unexpected {unexpected}")
    return skipped
