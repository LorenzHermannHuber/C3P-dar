"""Reference outputs of the original RadarGen inference (third_party/RadarGen) for test_radargen.py.

Runs the unchanged submodule code in an environment with RadarGen's requirements (mmcv, diffusers,
transformers; not the C3P-dar environment). The text encoder is loaded through the Hugging Face cache:
    HF_HOME=.cache/huggingface python backwards_comp/dump_radargen.py OUT.pt --frames A.h5 B.h5

Per frame, as RadarGenInference.get_pcl_from_bev_maps (batch 1) but with fixed noise: the transformed
condition maps, the noise, the encoded conditions, every DiT call (inputs, output), the decoder inputs
and outputs, the maps and the recovered cloud. Also the live empty-prompt Gemma embedding.
"""

import argparse
import sys
import time
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
ROOT = REPOSITORY / "third_party" / "RadarGen"
sys.path.insert(0, str(ROOT))
START = time.time()


def progress(message):
    print(f"[{time.time() - START:6.0f} s] {message}", flush=True)


progress(f"importing RadarGen from {ROOT}")

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from diffusion.data.transforms import get_transform  # noqa: E402
from diffusion.model.builder import get_vae  # noqa: E402
from radargen.inference.diffusion_pipeline import RadarGenDiffusionPipeline  # noqa: E402
from radargen.inference.pcl_recovery import recover_pcl_attributes  # noqa: E402

CONFIG = ROOT / "configs" / "RadarGen_600M_512px_TS_inference.yaml"
CHECKPOINT = REPOSITORY / "checkpoints" / "RadarGen_600M_512px_TS" / "RadarGen_600M_512px_TS.safetensors"
DCAE = REPOSITORY / "checkpoints" / "dc-ae-f32c32-sana-1.1"
CONDITIONS = ("appearance", "semantics", "velocity")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=0, help="seed of the noise")
    parser.add_argument("--fp32", action="store_true",
                        help="run the DiT in FP32 without TF32 (the DC-AE stays FP16, hard-coded in the pipeline)")
    args = parser.parse_args()
    if args.fp32:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False

    device = torch.device("cuda")
    progress(f"loading DC-AE from {DCAE}")
    vae = get_vae("dc-ae", str(DCAE), device).to(torch.float16)
    progress("building the pipeline (Gemma text encoder, DiT)")
    pipeline = RadarGenDiffusionPipeline(str(CONFIG), vae=vae, device=device)
    progress(f"loading {CHECKPOINT}")
    pipeline.from_pretrained(str(CHECKPOINT))
    if args.fp32:
        pipeline.weight_dtype = torch.float32
        pipeline.model.float()
    calls, decoded = [], []
    forward = pipeline.model.forward

    def recorded_forward(x, timestep, *args, **kwargs):
        output = forward(x, timestep, *args, **kwargs)
        calls.append(dict(x=x.cpu(), timestep=timestep.cpu(), output=output.cpu()))
        return output

    # The model's __call__ skips module hooks.
    pipeline.model.forward = recorded_forward
    pipeline.vae.decoder.register_forward_hook(lambda module, inputs, output: decoded.append(
        dict(input=inputs[0].cpu(), output=output.cpu())))
    image_size = pipeline.config.model.image_size
    coordinate_range = pipeline.config.data.extra["coordinate_range"]
    transform = get_transform("default_train_from_np", image_size)
    generator = torch.Generator().manual_seed(args.seed)

    frames = []
    for number, path in enumerate(args.frames, 1):
        progress(f"frame {number}/{len(args.frames)}: {path}")
        with h5py.File(path, "r") as f:
            maps = torch.stack([transform(f[f"conditioning/{name}"][()]) for name in CONDITIONS])
        noise = torch.randn(pipeline.num_maps, pipeline.config.vae.vae_latent_dim, *(2 * [pipeline.latent_size]),
                            generator=generator)
        conditions = [m[None].to(device) for m in maps]
        calls.clear()
        decoded.clear()
        with torch.no_grad():
            # Encodes the conditions in place.
            output = pipeline(prompt="", height_single_image=image_size, width_single_image=image_size,
                              guidance_scale=0.0, pag_guidance_scale=0.0, num_inference_steps=20,
                              latents=noise, bev_condition_maps=conditions)[0]
        radar_maps = np.stack([output.pd_map_np, output.rcs_map_np, output.doppler_map_np])
        locations, rcs, doppler = recover_pcl_attributes(*radar_maps, image_size, coordinate_range, device=device)
        frames.append(dict(
            path=str(path), maps=maps, noise=noise, conditions=torch.cat(conditions, dim=1).cpu(),
            calls=list(calls), decoder_inputs=torch.cat([d["input"] for d in decoded]),
            decoder_outputs=torch.cat([d["output"] for d in decoded]), radar_maps=torch.from_numpy(radar_maps),
            points=torch.from_numpy(np.column_stack([locations, rcs, doppler]).astype(np.float32))))
        progress(f"  {len(calls)} DiT calls, {len(locations)} points")

    token = pipeline.tokenizer("", max_length=pipeline.max_sequence_length, padding="max_length",
                               truncation=True, return_tensors="pt")
    torch.save(dict(frames=frames, fp32=args.fp32, text_embedding=pipeline.null_caption_embs[0].cpu(),
                    text_mask=token.attention_mask[0]), args.output)
    progress(f"wrote {args.output}")


if __name__ == "__main__":
    main()
