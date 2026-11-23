"""Reference outputs of the original PPP code for test_equivalence.py.

Runs in the environment of the original PPP repository, from that repository:
    cd <original repository>
    python <C3P-dar>/backwards_comp/dump_reference.py \
        evaluation/configs/PPP_validation_ss50_2867528_step20595_checkpoint_audit.yaml OUT.pt --frames A.h5 B.h5

Per frame (batch 1, as in the original validation): the bf16 fields of RadarGenPPPPipeline and the points
drawn with the generator seeded by SeedSequence([seed, position]). On the whole frame batch, in
eval mode: training_microbatch's losses and the per-parameter gradient norms of their sum.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pyrallis
import torch
import yaml

sys.path.insert(0, str(Path.cwd()))

from diffusion.data.transforms import get_transform  # noqa: E402
from diffusion.utils.config import SanaConfig  # noqa: E402
from radargen.inference.ppp_pipeline import RadarGenPPPPipeline  # noqa: E402
from radargen.losses import PPPLoss  # noqa: E402
from radargen.training.ppp_collate import collate_ppp_samples  # noqa: E402
from scripts.train_ppp import training_microbatch  # noqa: E402


class Autocast:
    def autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16)


def load_sample(path, transform, point_limit):
    with h5py.File(path, "r") as f:
        maps = {key: transform(f[f"conditioning/{name}"][()]) for name, key in
                (("appearance", "bev_color_map"), ("semantics", "bev_seg_map"), ("velocity", "bev_velocity_map"))}
        targets = f["ppp_targets"]
        detections = np.column_stack([targets[k][()] for k in ("xy", "rcs", "doppler")])
    return dict(maps, detections=torch.from_numpy(detections).float(), point_limit=point_limit)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("validation_config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", nargs="+", required=True)
    args = parser.parse_args()

    audit = yaml.safe_load(args.validation_config.read_text())
    with open(audit["model_config"]) as stream:
        config = pyrallis.load(SanaConfig, stream)
    torch.manual_seed(audit["seed"])
    pipeline = RadarGenPPPPipeline.from_config(
        config, checkpoint_path=audit["checkpoint_path"], sana_checkpoint_path=audit.get("sana_checkpoint_path"),
        null_embed_path=audit.get("null_embed_path"), text_model_dir=audit.get("text_model_dir"),
        device="cuda", seed=audit["seed"])
    point_limit = float(config.data.extra["coordinate_range"])
    transform = get_transform("default_train_from_np", config.model.image_size)
    samples = [load_sample(path, transform, point_limit) for path in args.frames]

    fields, points = [], []
    for position, sample in enumerate(samples):
        state = np.random.SeedSequence([audit["seed"], position]).generate_state(1)[0]
        pipeline.generator.manual_seed(int(state))
        prediction = pipeline.predict_fields(*(sample[k] for k in ("bev_color_map", "bev_seg_map", "bev_velocity_map")))
        fields.append(torch.stack([prediction[k] for k in ("log_intensity", "rcs_raw", "doppler_raw")]).cpu())
        points.append(pipeline.sample_fields(prediction))

    batch = collate_ppp_samples(samples)
    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    model = pipeline.model.eval()
    resolution = config.model.image_size * getattr(model, "output_scale", 1)
    ppp_loss = PPPLoss(out_img_size=(resolution, resolution),
                       normalize_by_gt_count=config.train.extra["normalize_ppp_by_gt_count"]).cuda()
    model.zero_grad(set_to_none=True)
    total, logs = training_microbatch(model, pipeline.condition_encoder, batch, pipeline.text_provider,
                                      ppp_loss, config, Autocast())
    total.backward()
    reference = dict(
        frames=list(args.frames), validation_config=str(args.validation_config.resolve()), seed=audit["seed"],
        fields=torch.stack(fields), points=points, losses={k: float(v) for k, v in logs.items()},
        grad_norms={name: p.grad.float().norm().item() for name, p in model.named_parameters() if p.grad is not None})
    if pipeline.text_provider is not None:
        y, mask = pipeline.text_provider(1, torch.device("cuda"))
        reference.update(text_embedding=y.float().cpu(), text_mask=mask.cpu())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(reference, args.output)
    print({k: round(v, 6) for k, v in reference["losses"].items()})


if __name__ == "__main__":
    main()
