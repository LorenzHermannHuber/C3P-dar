"""Convert a checkpoint of the original PPP training code into a C3P-dar run.

python backwards_comp/convert_checkpoint.py configs/dino_neck_compact.yaml \
    <original work dir>/train/2867528_ablation_dino_neck_compact_resume_4nodes/checkpoints/epoch_94_step_20595.pth \
    --name dino_neck_compact

Writes RUNS_ROOT/<name>/config.yaml and checkpoints/<checkpoint>.ckpt (weights, global step),
loadable with c3pdar.models.load_model. The model is loaded strictly.
DINO keys are identical. DiT: the frozen encoder comes from DC-AE, ``pos_embed`` is recomputed
(checked here), the single-token cross-attention keeps only the value half of ``kv_linear``,
and the text token is the committed empty-prompt token.
"""

import argparse
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file

from c3pdar import REPOSITORY, RUNS_ROOT
from c3pdar.models import build_model, dcae
from c3pdar.models.dit import _sincos_pos_embed


def convert_dit(model, state, pretrained):
    hidden = model.dit.x_embedder.proj.out_channels
    converted = {}
    for key, value in state.items():
        if key.startswith("ppp_decoder.decoder."):
            converted["decoder." + key.removeprefix("ppp_decoder.decoder.")] = value
        elif key == "dit.pos_embed":
            side = int(value.shape[1] ** 0.5)
            expected = torch.from_numpy(_sincos_pos_embed(hidden, side, side, model.dit.base_size)).float()
            print(f"pos_embed max |diff| to recomputed: {(value[0] - expected).abs().max().item():.3e}")
        elif key == "dit.y_embedder.y_embedding" or ".cross_attn.q_linear." in key:
            continue
        elif ".cross_attn.kv_linear." in key:
            converted[key.replace("kv_linear", "v_linear")] = value[hidden:]
        else:
            converted[key] = value
    converted["dit.text_token"] = load_file(str(REPOSITORY / pretrained["empty_prompt_path"]))["text_token"]
    if "dit.y_embedder.y_embedding" in state:
        diff = (state["dit.y_embedder.y_embedding"][:1] - converted["dit.text_token"]).abs().max().item()
        print(f"stored y_embedding vs empty-prompt token max |diff|: {diff:.3e}")
    dcae.load_pretrained(model.encoder, REPOSITORY / pretrained["dcae_path"], "encoder.")
    converted.update({f"encoder.{k}": v for k, v in model.encoder.state_dict().items()})
    return converted


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-root", type=Path, default=RUNS_ROOT)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    run_dir = args.output_root / args.name
    target = run_dir / "checkpoints" / f"{args.checkpoint.stem}.ckpt"
    if target.exists():
        raise SystemExit(f"{target} exists")

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    state = saved["state_dict"]
    model = build_model(config["model"], pretrained=False)
    if config["model"]["type"] == "dit":
        state = convert_dit(model, state, config["model"]["pretrained"])
    model.load_state_dict(state, strict=True)
    progress = saved.get("ppp_training", {}).get("progress", {})

    target.parent.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        if yaml.safe_load(config_path.read_text()) != config:
            raise SystemExit(f"{config_path} differs from {args.config}")
    else:
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    torch.save(dict(state_dict={f"model.{k}": v.contiguous() for k, v in model.state_dict().items()},
                    global_step=progress.get("global_step"), epoch=progress.get("epoch", saved.get("epoch")),
                    source=str(args.checkpoint.resolve())), target)
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
