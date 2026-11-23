"""Compare c3pdar.models.RadarGen with the original inference dumped by dump_radargen.py (GPU, C3P-dar environment).

python backwards_comp/test_radargen.py OUT.pt

Every stage runs on the reference inputs of that stage, so differences do not accumulate: dataset
normalization, condition encoding, each DiT call, the 20-step sampler, decoding, map conversion and
point recovery; finally the whole chain from the condition maps. Differences come from batching
(the original doubles the batch for guidance at scale 0) and GEMM/conv rounding in BF16/FP16.
A reference dumped with --fp32 runs both DiTs in FP32.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from c3pdar.data import load_frame
from c3pdar.evaluation import radargen
from c3pdar.models import RadarGen
from c3pdar.models import dcae
from c3pdar.models.radargen import COORDINATE_RANGE, DOPPLER_RANGE, RCS_RANGE


def report(name, ours, reference):
    ours, reference = torch.as_tensor(ours).double().cpu(), torch.as_tensor(reference).double().cpu()
    diff = (ours - reference).abs()
    scale = reference.abs().max().clamp_min(1e-12)
    print(f"  {name:<28} max {diff.max():.3e}  mean {diff.mean():.3e}  max/|ref| {diff.max() / scale:.3e}")


def compare_points(name, ours, reference):
    """Counts and nearest-neighbour distances (x,y in m) between the clouds; marks of matched points."""
    if not len(ours) or not len(reference):
        print(f"  {name:<28} {len(ours)} vs {len(reference)} points")
        return
    distance, nearest = cKDTree(reference[:, :2]).query(ours[:, :2])
    exact = distance < 1e-6
    marks = np.abs(ours[exact, 2:] - reference[nearest[exact], 2:]).max(initial=0)
    print(f"  {name:<28} {len(ours)} vs {len(reference)} points, {exact.sum()} at a reference location "
          f"(marks max diff {marks:.3e}), nearest max {distance.max():.3f} m")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()

    device = torch.device("cuda")
    print(f"loading {args.reference}", flush=True)
    reference = torch.load(args.reference, map_location="cpu", weights_only=False)
    print("loading RadarGen", flush=True)
    model = RadarGen.from_checkpoint(device=device)
    if reference.get("fp32"):
        print("FP32 reference: DiT in FP32 without TF32", flush=True)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model.dit.float()
    active = reference["text_mask"].nonzero().flatten()
    print(f"empty prompt: active tokens {active.tolist()}")
    report("text token vs live Gemma", model.dit.text_token.float(), reference["text_embedding"][active].float())

    for frame in reference["frames"]:
        print(frame["path"])
        maps = frame["maps"][None].to(device)
        report("normalized conditions", load_frame(frame["path"]).normalized_conditions()[0], frame["maps"])
        report("encoded conditions", model.encode(maps), frame["conditions"])

        conditions = frame["conditions"].to(device)
        # The original batch is [unconditional, conditional]; scale 0 keeps the first half.
        streams = len(frame["noise"])
        velocity = [model.dit(c["x"][:streams].to(device), c["timestep"][:streams].to(device), conditions)
                    for c in frame["calls"]]
        report(f"DiT ({len(velocity)} calls)", torch.stack(velocity).float(),
               torch.stack([c["output"][:streams] for c in frame["calls"]]).float())
        latents = model.denoise(conditions, frame["noise"][None].to(device))[0]
        report("denoised latents", latents.half() / dcae.SCALING_FACTOR, frame["decoder_inputs"])
        images = model.decoder(frame["decoder_inputs"].to(device))
        report("decoder", images, frame["decoder_outputs"])
        reference_images = torch.clamp(127.5 * frame["decoder_outputs"].to(device) + 128.0, 0, 255).float()
        radar_maps = radargen.radar_maps(reference_images.cpu().numpy(), RCS_RANGE, DOPPLER_RANGE)
        report("maps", radar_maps, frame["radar_maps"])
        points = radargen.recover_points(frame["radar_maps"].numpy(), COORDINATE_RANGE, device)
        compare_points("recovery", points, frame["points"].numpy())

        radar_maps = radargen.radar_maps(model(maps, frame["noise"][None].to(device))[0].cpu().numpy(),
                                         RCS_RANGE, DOPPLER_RANGE)
        for name, ours, ref in zip(("point density", "RCS", "Doppler"), radar_maps, frame["radar_maps"]):
            report(f"end to end: {name}", ours, ref)
        compare_points("end to end: points", radargen.recover_points(radar_maps, COORDINATE_RANGE, device),
                       frame["points"].numpy())


if __name__ == "__main__":
    main()
