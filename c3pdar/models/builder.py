"""Build a C3P-dar model from the ``model`` config section.

``type`` selects the architecture, the remaining keys are constructor arguments.
The optional ``pretrained`` mapping names the initialization weights of fresh
training (relative paths are relative to the repository); it is ignored when a
trained checkpoint is restored afterwards.
"""

from pathlib import Path

import torch
import yaml

from .. import REPOSITORY
from .dino import DinoPPP
from .dit import DiTPPP
from .resnet import ResNetPPP


def build_model(cfg, pretrained=True):
    cfg = dict(cfg)
    kind = cfg.pop("type")
    weights = {k: str(REPOSITORY / v) if v else v for k, v in (cfg.pop("pretrained", None) or {}).items()}
    if kind == "dino":
        return DinoPPP(**cfg, **(weights if pretrained else {}))
    if kind == "resnet":
        return ResNetPPP(**cfg, **(weights if pretrained else {}))
    if kind == "dit":
        model = DiTPPP(**cfg)
        if pretrained:
            model.load_pretrained(**weights)
        return model
    raise ValueError(f"Unknown model type {kind!r}")


def load_model(checkpoint, config=None, device="cpu"):
    """Trained model, config (default <run>/config.yaml) and global step of a run checkpoint."""
    checkpoint = Path(checkpoint)
    if config is None:
        config = yaml.safe_load((checkpoint.parent.parent / "config.yaml").read_text())
    model = build_model(config["model"], pretrained=False)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    state = saved["state_dict"]
    model.load_state_dict({k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")})
    return model.to(device).eval(), config, saved.get("global_step")
