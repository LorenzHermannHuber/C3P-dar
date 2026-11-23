"""DINO C3P-dar model: DINOv3 ViT with per-modality LoRA, RoPE transformer neck, attention head.

The three condition maps are encoded independently by one backbone (a separate LoRA
per modality), concatenated along channels, refined by the neck and decoded to
(B,3,H*s,W*s) fields [log-intensity, raw RCS, raw Doppler].
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .heads import AttentionPixelShuffleHead


class LoRALinear(nn.Module):
    """Base linear plus one low-rank update per modality.

    Inputs are grouped as [appearance batch, semantics batch, velocity batch];
    each contiguous chunk uses its own adapter.
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float, dropout: float, num_adapters: int = 3):
        super().__init__()
        self.linear = linear
        self.scale = alpha / rank
        self.num_adapters = num_adapters
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.ModuleList(nn.Linear(linear.in_features, rank, bias=False) for _ in range(num_adapters))
        self.lora_B = nn.ModuleList(nn.Linear(rank, linear.out_features, bias=False) for _ in range(num_adapters))
        for lora_a, lora_b in zip(self.lora_A, self.lora_B):
            nn.init.kaiming_uniform_(lora_a.weight, a=math.sqrt(5))
            nn.init.zeros_(lora_b.weight)

    def forward(self, x):
        updates = [b(a(self.dropout(chunk)))
                   for a, b, chunk in zip(self.lora_A, self.lora_B, x.chunk(self.num_adapters, dim=0))]
        return self.linear(x) + self.scale * torch.cat(updates, dim=0)


def _rope_grid(height, width, head_dim, device):
    """2D rotary angles for the patch grid, half of the head dimension per axis."""
    rows, cols = torch.meshgrid(torch.arange(height, device=device, dtype=torch.float32),
                                torch.arange(width, device=device, dtype=torch.float32), indexing="ij")
    axis_dim = head_dim // 2
    frequencies = 2000.0 ** (-torch.arange(0, axis_dim, 2, device=device).float() / axis_dim)
    angles = torch.cat([rows.flatten()[:, None] * frequencies, cols.flatten()[:, None] * frequencies], dim=-1)
    return angles.cos()[None, None], angles.sin()[None, None]


def _apply_rope(x, cos, sin):
    first, second = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    rotated = torch.stack([first * cos - second * sin, first * sin + second * cos], dim=-1)
    return rotated.flatten(-2).to(x.dtype)


def _rms_norm(x, scale):
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    return (normalized * scale).to(x.dtype)


class ParallelTransformerBlock(nn.Module):
    """Attention and MLP share one normalized input; one fused input and output projection."""

    def __init__(self, dim, num_heads, head_dim, mlp_hidden_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attention_dim = num_heads * head_dim
        self.mlp_hidden = mlp_hidden_dim
        self.pre_norm = nn.LayerNorm(dim, bias=False)
        self.linear1 = nn.Linear(dim, 3 * self.attention_dim + mlp_hidden_dim, bias=False)
        self.linear2 = nn.Linear(self.attention_dim + mlp_hidden_dim, dim, bias=False)
        self.q_scale = nn.Parameter(torch.ones(head_dim))
        self.k_scale = nn.Parameter(torch.ones(head_dim))
        self.mlp_act = nn.SELU()

    def forward(self, x, cos, sin):
        batch, length, _ = x.shape
        qkv, hidden = self.linear1(self.pre_norm(x)).split([3 * self.attention_dim, self.mlp_hidden], dim=-1)
        q, k, v = qkv.reshape(batch, length, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q = _apply_rope(_rms_norm(q, self.q_scale), cos, sin)
        k = _apply_rope(_rms_norm(k, self.k_scale), cos, sin)
        attended = F.scaled_dot_product_attention(q, k, v)
        attended = attended.transpose(1, 2).reshape(batch, length, self.attention_dim)
        return x + self.linear2(torch.cat([attended, self.mlp_act(hidden)], dim=-1))


class RoPETransformerNeck(nn.Module):
    """Refine (B,D,h,w) patch features without changing their grid."""

    def __init__(self, dim, num_layers, num_heads, head_dim=None, mlp_hidden_dim=None, mlp_ratio=4.0,
                 grad_checkpointing=False):
        super().__init__()
        head_dim = head_dim or dim // num_heads
        mlp_hidden_dim = mlp_hidden_dim or int(dim * mlp_ratio)
        if head_dim % 4:
            raise ValueError("Neck head dimension must be divisible by four for 2D RoPE")
        self.head_dim = head_dim
        self.grad_checkpointing = grad_checkpointing
        self.layers = nn.ModuleList(ParallelTransformerBlock(dim, num_heads, head_dim, mlp_hidden_dim)
                                    for _ in range(num_layers))
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x):
        batch, channels, height, width = x.shape
        cos, sin = _rope_grid(height, width, self.head_dim, x.device)
        tokens = x.flatten(2).transpose(1, 2)
        for layer in self.layers:
            if self.grad_checkpointing and self.training and torch.is_grad_enabled():
                tokens = checkpoint(layer, tokens, cos, sin, use_reentrant=False)
            else:
                tokens = layer(tokens, cos, sin)
        return self.out_norm(tokens).transpose(1, 2).reshape(batch, channels, height, width)


class DinoPPP(nn.Module):
    """(B,3,H,W) x3 condition maps in [-1,1] -> (B,3,H*s,W*s) [log-intensity, raw RCS, raw Doppler].

    ``checkpoint_path`` loads pretrained DINOv3 weights (timm); empty means random init.
    """

    def __init__(self, model_name="vit_small_patch16_dinov3", checkpoint_path="", frozen_base=False,
                 lora_r=16, lora_alpha=32.0, lora_dropout=0.0,
                 target_modules=("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"),
                 neck_layers=12, neck_heads=4, neck_head_dim=None, neck_mlp_hidden_dim=None, mlp_ratio=4.0,
                 head=None, grad_checkpointing=False):
        super().__init__()
        import timm

        kwargs = dict(pretrained=False, in_chans=3, num_classes=0)
        if checkpoint_path:
            kwargs["checkpoint_path"] = str(checkpoint_path)
        self.backbone = timm.create_model(model_name, **kwargs)
        self.frozen_base = frozen_base
        self.backbone.requires_grad_(not frozen_base)
        for name, module in list(self.backbone.named_modules()):
            if isinstance(module, nn.Linear) and any(name.endswith(t) for t in target_modules):
                parent, _, child = name.rpartition(".")
                setattr(self.backbone.get_submodule(parent), child,
                        LoRALinear(module, lora_r, lora_alpha, lora_dropout))
        if grad_checkpointing:
            self.backbone.set_grad_checkpointing(True)

        dim = 3 * self.backbone.embed_dim
        self.neck = RoPETransformerNeck(dim, neck_layers, neck_heads, neck_head_dim, neck_mlp_hidden_dim,
                                        mlp_ratio, grad_checkpointing)
        self.head = AttentionPixelShuffleHead(dim, grad_checkpointing=grad_checkpointing, **(head or {}))
        self.output_scale = self.head.output_scale
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen_base:
            # Pretrained stochastic layers stay off; LoRA dropout follows the mode.
            self.backbone.eval()
            for module in self.backbone.modules():
                if isinstance(module, LoRALinear):
                    module.train(mode)
        return self

    def forward(self, bev_color_map, bev_seg_map, bev_velocity_map):
        height, width = bev_color_map.shape[-2:]
        batch_size = bev_color_map.shape[0]
        # Modality-major batch order routes each modality to its own LoRA.
        x = torch.cat([bev_color_map, bev_seg_map, bev_velocity_map], dim=0)
        # Bottom/right padding to whole patches; cropped again below.
        x = F.pad(x, (0, -width % 16, 0, -height % 16))
        (features,) = self.backbone.forward_intermediates(
            x, indices=[len(self.backbone.blocks) - 1], norm=True, output_fmt="NCHW", intermediates_only=True)
        x = torch.cat(features.split(batch_size, dim=0), dim=1)
        x = self.head(self.neck(x))
        return x[..., :height * self.output_scale, :width * self.output_scale]
