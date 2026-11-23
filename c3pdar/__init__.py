"""C3P-dar. Default locations of data, weights, runs and evaluation outputs; environment variables override them."""

import os
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("C3PDAR_DATA_ROOT", "/e/scratch/nxtaim-1/public/truckscenes/RadarGen_preprocessed"))
# Pretrained weights (scripts/download_checkpoints.py) and the committed empty-prompt token.
CHECKPOINTS = Path(os.environ.get("C3PDAR_CHECKPOINTS", REPOSITORY / "checkpoints"))
RADARGEN_CHECKPOINT = CHECKPOINTS / "RadarGen_600M_512px_TS" / "RadarGen_600M_512px_TS.safetensors"
DCAE = CHECKPOINTS / "dc-ae-f32c32-sana-1.1"
EMPTY_PROMPT = CHECKPOINTS / "empty_prompt_gemma-2-2b-it.safetensors"
# VoxelNeXt radar detector (not public, not downloadable) and the TruckScenes devkit tables it is scored with.
VOXELNEXT = CHECKPOINTS / "voxelnext_truckscenes"
TRUCKSCENES_ROOT = Path(os.environ.get("C3PDAR_TRUCKSCENES_ROOT", "/e/scratch/nxtaim-1/public/truckscenes/man-truckscenes"))
# Training runs: RUNS_ROOT/<name>/{config.yaml, checkpoints, logs, tensorboard}.
RUNS_ROOT = Path(os.environ.get("C3PDAR_RUNS_ROOT", REPOSITORY / "outputs"))
# Raw outputs per model (EVAL_RAW/<name>/<scene_token>.h5) and metric tables (EVAL_METRICS/<name>/<selection>/).
EVAL_RAW = Path(os.environ.get("C3PDAR_EVAL_RAW", REPOSITORY / ".eval_raw"))
EVAL_METRICS = Path(os.environ.get("C3PDAR_EVAL_METRICS", REPOSITORY / ".eval_metrics"))
