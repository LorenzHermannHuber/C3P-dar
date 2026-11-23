"""Compare a converted C3P-dar checkpoint with the original PPP code's reference dump (dump_reference.py).

python backwards_comp/test_equivalence.py RUN/checkpoints/X.ckpt REF.pt

Same frames, same eval-mode bf16 forward: fields (batch 1), the sampler on the reference fields
(must be identical), the losses of the frame batch and per-parameter gradient norms.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from c3pdar.data import RadarGenBatch, load_frame
from c3pdar.loss import PPPLoss, weighted_loss
from c3pdar.models import load_model
from c3pdar.sampling import cell_masses, physical_marks, sample_generator, sample_points


def reference_name(name):
    """C3P-dar parameter name -> the original name."""
    if name.startswith("decoder."):
        return "ppp_decoder." + name
    return name.replace(".cross_attn.v_linear.", ".cross_attn.kv_linear.")


def report(name, ours, theirs):
    ours, theirs = torch.as_tensor(ours, dtype=torch.float64), torch.as_tensor(theirs, dtype=torch.float64)
    diff = (ours - theirs).abs()
    scale = theirs.abs().max().clamp_min(1e-12)
    print(f"{name:40s} max|diff| {diff.max():.3e}  mean|diff| {diff.mean():.3e}  rel {diff.max() / scale:.3e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()

    reference = torch.load(args.reference, map_location="cpu", weights_only=False)
    device = torch.device("cuda")
    model, config, _ = load_model(args.checkpoint, device=device)
    point_limit = 50.0
    samples = [load_frame(path, position) for position, path in enumerate(reference["frames"])]

    if "text_embedding" in reference:
        active = reference["text_mask"].flatten().nonzero().flatten()
        report("text token (live Gemma)", model.dit.text_token.cpu(), reference["text_embedding"].flatten(0, -2)[active])

    with torch.no_grad():
        for position, sample in enumerate(samples):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                fields = model(*sample.to(device).normalized_conditions().unbind(1))[0].float().cpu()
            theirs = reference["fields"][position]
            for channel, name in enumerate(("log_intensity", "rcs_raw", "doppler_raw")):
                report(f"frame {position} {name}", fields[channel], theirs[channel])
            masses = cell_masses(theirs[0])
            marks = [physical_marks(theirs[c], config["loss"]["mark_link"]) for c in (1, 2)]
            points = sample_points(masses, *marks, point_limit, sample_generator(reference["seed"], position))
            expected = reference["points"][position]
            same = points.shape == expected.shape and np.array_equal(points, expected)
            print(f"frame {position} sampler on reference fields identical: {same} "
                  f"({len(points)} vs {len(expected)} points)")

    batch = RadarGenBatch.collate(samples).to(device)
    model.zero_grad(set_to_none=True)
    loss_cfg = dict(config["loss"])
    weights = loss_cfg.pop("weights")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        fields = model(*batch.normalized_conditions().unbind(1))
        losses = PPPLoss(**loss_cfg)(fields, batch)
    total, logs = weighted_loss(losses, weights)
    total.backward()
    for key in ("ppp", "rcs", "doppler"):
        print(f"loss {key:8s} ours {logs[f'loss_{key}'].item():.6f}  "
              f"reference {reference['losses'][f'loss_{key}_raw']:.6f}")
    print(f"loss total    ours {logs['loss_total'].item():.6f}  reference {reference['losses']['loss_total']:.6f}")

    ours = {name: p.grad.float().norm().item() for name, p in model.named_parameters() if p.grad is not None}
    pairs = [(name, value, reference["grad_norms"].get(reference_name(name))) for name, value in ours.items()]
    missing = [name for name, _, theirs in pairs if theirs is None]
    pairs = [(n, o, t) for n, o, t in pairs if t is not None]
    rel = np.array([abs(o - t) / max(t, 1e-12) for _, o, t in pairs])
    print(f"gradient norms: {len(pairs)} parameters compared, {len(missing)} without reference "
          f"{missing[:5]}; relative difference median {np.median(rel):.3e}, max {rel.max():.3e} "
          f"({pairs[int(rel.argmax())][0]})")


if __name__ == "__main__":
    main()
