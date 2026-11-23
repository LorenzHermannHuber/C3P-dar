"""The original RadarGen baseline: SANA-600M flow-matching DiT on DC-AE latents, sampled with DPM-Solver++.

Port of third_party/RadarGen's inference (RadarGenDiffusionPipeline, configs/RadarGen_600M_512px_TS_inference.yaml)
for its released checkpoint: DC-AE in FP16 (encoding under FP16 autocast), DiT in BF16 without FP32
attention, 20 steps of 2nd-order multistep DPM-Solver++ on the shifted flow schedule. Each frame's three
maps (point density, RCS, Doppler) are denoised jointly as three streams of the dit.py blocks, each stream
being [noisy latent, three condition latents, modality embedding]. As in dit.py the empty prompt has a
single active token, so cross-attention is the bias proj(v(y)). RadarGen samples with classifier-free
guidance at scale 0, i.e. the unconditional prediction; with an empty prompt on both sides it is computed
once instead of on a doubled batch. backwards_comp/test_radargen.py checks the port against the original.
"""

import torch
from torch import nn

from .. import DCAE, EMPTY_PROMPT, RADARGEN_CHECKPOINT
from . import dcae
from .dit import NUM_STREAMS, CaptionEmbedder, DiTBlock, PatchEmbed, RMSNorm, TimestepEmbedder, t2i_modulate

LATENT_CHANNELS = 32
STEPS = 20
FLOW_SHIFT = 3.0
# Map decoding and point recovery (data.extra of the inference config).
RCS_RANGE = (-20, 66)
DOPPLER_RANGE = (-120, 120)
COORDINATE_RANGE = 50


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.scale_shift_table = nn.Parameter(torch.zeros(2, hidden_size))

    def forward(self, x, t):
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        return self.linear(t2i_modulate(self.norm_final(x), shift, scale))


class RadarGenDiT(nn.Module):
    """Velocity of the (3B,32,h,w) noisy latents at model time (3B,) in [0,1000], given (B,96,h,w) conditions."""

    def __init__(self, depth=28, hidden_size=1152, mlp_ratio=2.5, head_dim=32, caption_channels=2304,
                 latent_channels=LATENT_CHANNELS, y_norm_scale_factor=0.01, norm_eps=1e-5):
        super().__init__()
        self.modality_pe = nn.Parameter(torch.zeros(NUM_STREAMS, 3))
        self.x_embedder = PatchEmbed((1 + NUM_STREAMS) * latent_channels + 3, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        self.y_embedder = CaptionEmbedder(caption_channels, hidden_size)
        self.attention_y_norm = RMSNorm(hidden_size, scale_factor=y_norm_scale_factor, eps=norm_eps)
        self.register_buffer("text_token", torch.zeros(1, caption_channels))
        self.blocks = nn.ModuleList(DiTBlock(hidden_size, mlp_ratio, head_dim, norm_eps, fp32_attention=False,
                                             qk_norm=False) for _ in range(depth))
        self.final_layer = FinalLayer(hidden_size, latent_channels)

    def forward(self, x, timestep, conditions):
        batch, _, height, width = conditions.shape
        x = x.reshape(batch, NUM_STREAMS, -1, height, width)
        streams = conditions[:, None].expand(-1, NUM_STREAMS, -1, -1, -1)
        modality = self.modality_pe[None, :, :, None, None].expand(batch, -1, -1, height, width)
        # The concatenation promotes to FP32 before the cast to the weight dtype, as in the original.
        x = torch.cat((x, streams, modality), dim=2).flatten(0, 1)
        dtype = self.x_embedder.proj.weight.dtype
        x = self.x_embedder(x.to(dtype))
        # The original casts the timestep to the weight dtype too (BF16: e.g. 999.67 -> 1000).
        t = self.t_embedder(timestep.to(dtype))
        t0 = self.t_block(t)
        y = self.attention_y_norm(self.y_embedder.y_proj(self.text_token.to(dtype)))
        for block in self.blocks:
            x = block(x, y, t0, (height, width))
        x = self.final_layer(x, t)
        return x.transpose(1, 2).reshape(x.shape[0], -1, height, width)


def flow_timesteps(steps=STEPS, shift=FLOW_SHIFT, device=None):
    """(steps+1,) descending t from ~1 to 0: DPM_Solver.get_time_steps("time_uniform_flow", t_T=1, t_0=1e-3)."""
    sigmas = 1.0 - torch.linspace(1, 1e-3, steps + 1).to(device)
    return (shift * sigmas / (1 + (shift - 1) * sigmas)).flip(0)


def dpm_solver(data_prediction, x, steps=STEPS, shift=FLOW_SHIFT):
    """Multistep DPM-Solver++ of order 2 (solver_type "dpmsolver", lower_order_final) for alpha=1-t, sigma=t.

    ``data_prediction(x, t)`` is the x0 estimate at the 0-dim time t. The first and the last step are
    first order; the model is not evaluated at t=0. Same operations as DPM_Solver.sample(method="multistep").
    """
    timesteps = flow_timesteps(steps, shift, x.device)

    def log_snr(t):
        return torch.log(1 - t) - torch.log(t)

    previous = [(timesteps[0], data_prediction(x, timesteps[0]))]
    for step in range(1, steps + 1):
        t = timesteps[step]
        s, model_s = previous[-1]
        h = log_snr(t) - log_snr(s)
        alpha_t = torch.exp(torch.log(1 - t))
        phi_1 = torch.expm1(-h)
        if step == 1 or step == steps:
            x = t / s * x - alpha_t * phi_1 * model_s
        else:
            s_1, model_s_1 = previous[-2]
            r0 = (log_snr(s) - log_snr(s_1)) / h
            d1 = (1.0 / r0) * (model_s - model_s_1)
            x = (t / s) * x - (alpha_t * phi_1) * model_s - 0.5 * (alpha_t * phi_1) * d1
        if step < steps:
            previous = [previous[-1], (t, data_prediction(x, t))]
    return x


class RadarGen(nn.Module):
    """(B,3,3,H,W) condition maps in [-1,1] -> (B,3,3,H,W) decoded RGB images in [0,255] of
    [point density, RCS, Doppler]; c3pdar.evaluation.radargen turns them into maps and a point cloud."""

    def __init__(self, **dit):
        super().__init__()
        self.encoder = dcae.Encoder()
        self.dit = RadarGenDiT(**dit)
        self.decoder = dcae.Decoder(LATENT_CHANNELS, 3)

    @classmethod
    def from_checkpoint(cls, checkpoint=RADARGEN_CHECKPOINT, empty_prompt=EMPTY_PROMPT, dcae_path=DCAE, device="cpu"):
        """The released model with the original dtypes: FP16 DC-AE, BF16 DiT."""
        from safetensors.torch import load_file

        model = cls()
        hidden = model.dit.x_embedder.proj.out_channels
        state = {"text_token": load_file(str(empty_prompt))["text_token"]}
        for key, value in load_file(str(checkpoint)).items():
            # pos_embed is unused (use_pe false), y_embedding is the training-time caption dropout embedding.
            if key in ("pos_embed", "y_embedder.y_embedding") or ".q_linear." in key:
                continue
            if ".cross_attn.kv_linear." in key:
                key, value = key.replace("kv_linear", "v_linear"), value[hidden:]
            state[key] = value
        model.dit.load_state_dict(state)
        for module, prefix in ((model.encoder, "encoder."), (model.decoder, "decoder.")):
            if dcae.load_pretrained(module, dcae_path, prefix):
                raise KeyError(f"DC-AE {prefix} shapes do not match")
        model.encoder.half()
        model.decoder.half()
        model.dit.to(torch.bfloat16)
        return model.to(device).eval().requires_grad_(False)

    def encode(self, maps):
        """(B,3,3,H,W) -> (B,96,h,w) FP16 scaled latents of [appearance, semantics, velocity]."""
        with torch.autocast(maps.device.type, dtype=torch.float16):
            return torch.cat([self.encoder(m) * dcae.SCALING_FACTOR for m in maps.unbind(1)], dim=1)

    def denoise(self, conditions, noise, steps=STEPS):
        """(B,3,32,h,w) FP32 noise -> (B,3,32,h,w) FP32 latents of the three maps."""
        x = noise.flatten(0, 1)

        def data_prediction(x, t):
            velocity = self.dit(x, t.expand(len(x)) * 1000, conditions).float()
            epsilon = (1 - t) * velocity + x
            return (x - t * epsilon) / (1 - t)

        return dpm_solver(data_prediction, x, steps).reshape(noise.shape)

    def decode(self, latents):
        """(..., 32, h, w) latents -> (..., 3, 32h, 32w) FP32 images in [0,255], rounded in FP16 as the original."""
        images = self.decoder(latents.flatten(0, -4).half() / dcae.SCALING_FACTOR)
        images = torch.clamp(127.5 * images + 128.0, 0, 255).float()
        return images.reshape(*latents.shape[:-3], *images.shape[1:])

    @torch.no_grad()
    def forward(self, maps, noise, steps=STEPS):
        return self.decode(self.denoise(self.encode(maps), noise, steps))
