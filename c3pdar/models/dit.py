"""DiT C3P-dar model: frozen DC-AE encoder, SANA-600M linear-attention DiT, full-width DC-AE decoder.

The DiT is not used as a diffusion model: the timestep is fixed at zero and there is
no noisy input. Each frame becomes three streams (one per output field) that share the
condition latents and differ by a learned modality embedding; self-attention runs
jointly over the three streams. The text conditioning is SANA's empty prompt, which
has a single active token, so cross-attention reduces to a learned per-block bias
proj(v(y)). The decoder maps every stream's 1152-channel features to one field.

Key names follow SANA (``dit.*``) and DC-AE (``encoder.*``, ``decoder.*``).
"""

from functools import lru_cache
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from . import dcae

NUM_STREAMS = 3


class RMSNorm(nn.Module):
    def __init__(self, dim, scale_factor=1.0, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim) * scale_factor)

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * normalized).type_as(x)


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


@lru_cache
def _sincos_pos_embed(dim, height, width, base_size):
    """SANA's fixed 2D sin-cos embedding (float64 numpy, (H*W, dim))."""
    grid_h = np.arange(height, dtype=np.float32) / (height / base_size)
    grid_w = np.arange(width, dtype=np.float32) / (width / base_size)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0).reshape(2, 1, width, height)

    def embed_1d(pos, half):
        omega = 1.0 / 10000 ** (np.arange(half // 2, dtype=np.float64) / (half / 2.0))
        out = np.einsum("m,d->md", pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    return np.concatenate([embed_1d(grid[0], dim // 2), embed_1d(grid[1], dim // 2)], axis=1)


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_size, 1, bias=True)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size), nn.SiLU(),
                                 nn.Linear(hidden_size, hidden_size))
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(embedding.to(self.mlp[0].weight.dtype))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class CaptionEmbedder(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super().__init__()
        self.y_proj = Mlp(in_channels, hidden_size, hidden_size)


class LiteLA(nn.Module):
    """ReLU linear self-attention, by default with RMS-normalized queries and keys."""

    def __init__(self, dim, head_dim=32, eps=1e-8, norm_eps=1e-5, fp32_attention=True, qk_norm=True):
        super().__init__()
        self.head_dim = head_dim
        self.eps = eps
        self.fp32_attention = fp32_attention
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(dim, eps=norm_eps) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(dim, eps=norm_eps) if qk_norm else nn.Identity()

    @torch.autocast("cuda", enabled=False)
    def attn_matmul(self, q, k, v):
        q, k = F.relu(q), F.relu(k)
        if self.fp32_attention:
            q, k, v = q.float(), k.float(), v.float()
        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=1)
        out = torch.matmul(torch.matmul(v, k), q)
        if out.dtype in (torch.float16, torch.bfloat16):
            out = out.float()
        return out[:, :, :-1] / (out[:, :, -1:] + self.eps)

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, c).unbind(2)
        dtype = q.dtype
        heads = c // self.head_dim
        q = self.q_norm(q).transpose(-1, -2).reshape(b, heads, self.head_dim, n)
        k = self.k_norm(k).transpose(-1, -2).reshape(b, heads, self.head_dim, n).transpose(-1, -2)
        v = v.transpose(-1, -2).reshape(b, heads, self.head_dim, n)
        out = self.attn_matmul(q, k, v).to(dtype)
        return self.proj(out.view(b, c, n).permute(0, 2, 1))


class TextBias(nn.Module):
    """Cross-attention to a single text token: softmax weight 1, so the output is proj(v(y))."""

    def __init__(self, dim):
        super().__init__()
        self.v_linear = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, y):
        return self.proj(self.v_linear(y))


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, groups=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, groups=groups,
                              bias=bias)

    def forward(self, x):
        return self.conv(x)


class GLUMBConv(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.inverted_conv = ConvLayer(dim, 2 * hidden)
        self.depth_conv = ConvLayer(2 * hidden, 2 * hidden, 3, groups=2 * hidden)
        self.point_conv = ConvLayer(hidden, dim, bias=False)

    def forward(self, x, hw):
        b, n, c = x.shape
        x = x.reshape(b, *hw, c).permute(0, 3, 1, 2)
        x, gate = self.depth_conv(F.silu(self.inverted_conv(x))).chunk(2, dim=1)
        return self.point_conv(x * F.silu(gate)).reshape(b, c, n).permute(0, 2, 1)


class DiTBlock(nn.Module):
    """adaLN-modulated joint linear self-attention over the three streams, text bias, GLU conv MLP."""

    def __init__(self, dim, mlp_ratio, head_dim, norm_eps, fp32_attention, qk_norm=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = LiteLA(dim, head_dim, norm_eps=norm_eps, fp32_attention=fp32_attention, qk_norm=qk_norm)
        self.cross_attn = TextBias(dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GLUMBConv(dim, int(dim * mlp_ratio))
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim ** 0.5)

    def forward(self, x, y, t, hw):
        b, n, c = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(b, 6, -1)).chunk(6, dim=1)
        # Joint attention over the streams of a frame: (b*3, n, c) -> (b, 3*n, c).
        attended = self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa).reshape(b // NUM_STREAMS, -1, c))
        x = x + gate_msa * attended.reshape(b, n, c)
        x = x + self.cross_attn(y)
        return x + gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp), hw)


class DiT(nn.Module):
    """Three (B,32,h,w) scaled condition latents -> (3B,hidden,h,w) per-stream features."""

    def __init__(self, depth=28, hidden_size=1152, mlp_ratio=2.5, head_dim=32, caption_channels=2304,
                 latent_channels=32, base_size=16, y_norm_scale_factor=0.01, norm_eps=1e-5,
                 fp32_attention=True, grad_checkpointing=False):
        super().__init__()
        self.base_size = base_size
        self.grad_checkpointing = grad_checkpointing
        self.modality_pe = nn.Parameter(torch.zeros(NUM_STREAMS, 3))
        self.x_embedder = PatchEmbed(NUM_STREAMS * latent_channels + 3, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        self.y_embedder = CaptionEmbedder(caption_channels, hidden_size)
        self.attention_y_norm = RMSNorm(hidden_size, scale_factor=y_norm_scale_factor, eps=norm_eps)
        # SANA's empty-prompt Gemma embedding of the single active token.
        self.register_buffer("text_token", torch.zeros(1, caption_channels))
        self.blocks = nn.ModuleList(DiTBlock(hidden_size, mlp_ratio, head_dim, norm_eps, fp32_attention)
                                    for _ in range(depth))
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.x_embedder.proj.weight.flatten(1))
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)
        nn.init.normal_(self.modality_pe, std=0.1)

    def forward(self, conditions):
        batch, _, height, width = conditions[0].shape
        streams = torch.cat(conditions, dim=1)[:, None].expand(-1, NUM_STREAMS, -1, -1, -1)
        modality = self.modality_pe[None, :, :, None, None].expand(batch, -1, -1, height, width)
        x = torch.cat((streams, modality), dim=2).reshape(batch * NUM_STREAMS, -1, height, width)
        x = self.x_embedder(x.float())
        pos = _sincos_pos_embed(x.shape[-1], height, width, self.base_size)
        x = (x.float() + torch.from_numpy(pos).to(x.device, torch.float32)).to(x.dtype)
        t = self.t_embedder(torch.zeros(x.shape[0], device=x.device))
        t0 = self.t_block(t)
        y = self.attention_y_norm(self.y_embedder.y_proj(self.text_token))
        for block in self.blocks:
            if self.grad_checkpointing and torch.is_grad_enabled():
                x = checkpoint(block, x, y, t0, (height, width), use_reentrant=False)
            else:
                x = block(x, y, t0, (height, width))
        return x.transpose(1, 2).reshape(x.shape[0], -1, height, width)


class DiTPPP(nn.Module):
    """(B,3,H,W) x3 condition maps in [-1,1] -> (B,3,H,W) [log-intensity, raw RCS, raw Doppler]."""

    output_scale = 1

    def __init__(self, image_size=512, grad_checkpointing=False, decoder_grad_checkpointing=None, **dit):
        super().__init__()
        self.encoder = dcae.Encoder().requires_grad_(False).eval()
        self.dit = DiT(base_size=image_size // 32, grad_checkpointing=grad_checkpointing, **dit)
        if decoder_grad_checkpointing is None:
            decoder_grad_checkpointing = grad_checkpointing
        self.decoder = dcae.Decoder(self.dit.x_embedder.proj.out_channels, 1,
                                    gradient_checkpointing=decoder_grad_checkpointing)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, bev_color_map, bev_seg_map, bev_velocity_map):
        batch = bev_color_map.shape[0]
        with torch.no_grad():
            conditions = [self.encoder(x.float()) * dcae.SCALING_FACTOR
                          for x in (bev_color_map, bev_seg_map, bev_velocity_map)]
        features = self.dit(conditions)
        if torch.is_autocast_enabled(features.device.type):
            features = features.to(torch.get_autocast_dtype(features.device.type))
        fields = self.decoder(features)
        return fields.reshape(batch, NUM_STREAMS, *fields.shape[-2:])

    def load_pretrained(self, sana_path, empty_prompt_path, dcae_path):
        """Fresh-training initialization from SANA-600M, the empty-prompt token and DC-AE."""
        from safetensors.torch import load_file

        state = torch.load(sana_path, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        hidden = self.dit.x_embedder.proj.out_channels
        converted = {"text_token": load_file(str(empty_prompt_path))["text_token"]}
        for key, value in state.items():
            if key.startswith(("pos_embed", "final_layer.", "y_embedder.y_embedding")) or ".q_linear." in key:
                continue
            if ".cross_attn.kv_linear." in key:
                key, value = key.replace("kv_linear", "v_linear"), value[hidden:]
            converted[key] = value
        old = converted.pop("x_embedder.proj.weight")
        missing, unexpected = self.dit.load_state_dict(converted, strict=False)
        allowed = {"modality_pe", "x_embedder.proj.weight"}
        allowed |= {k for k in self.dit.state_dict() if ".attn.q_norm." in k or ".attn.k_norm." in k}
        if unexpected or set(missing) - allowed:
            raise KeyError(f"SANA mismatch: missing {missing}, unexpected {unexpected}")
        with torch.no_grad():
            # Fresh xavier init (from initialize) for the modality channels.
            self.dit.x_embedder.proj.weight[:, :old.shape[1] * NUM_STREAMS].copy_(old.repeat(1, NUM_STREAMS, 1, 1)
                                                                                  / NUM_STREAMS)
        dcae.load_pretrained(self.encoder, dcae_path, "encoder.")
        dcae.load_pretrained(self.decoder, dcae_path, "decoder.")
