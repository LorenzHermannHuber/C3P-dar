# Refactor plan

Goal: replace the student code (original PPP repository @a307139) with a small, readable
repository that loads the trained models, trains them, and evaluates them uniformly.

## Findings that shape the plan

- Training backend: `scripts/train_ppp.py` (~900 lines) uses HF Accelerate (DDP, bf16
  autocast), launched via `torch.distributed.run`; config through pyrallis `SanaConfig`.
- All three selected models are deterministic one-pass predictors of three dense fields
  (log-intensity, RCS, Doppler). Radar point clouds are drawn from these fields by
  Poisson sampling, which is cheap. A sampling budget rescales the intensity to a target
  expected point count.
- DiT DC-AE (run 2867524): diffusion is not used. The timestep is fixed at 0 and the text
  input is the Gemma encoding of the empty prompt, which is the same for every sample. It
  can be stored once as a buffer, which removes Gemma, transformers and the diffusion
  schedulers.
  - The frozen DC-AE encoder for the conditions is not in the training checkpoint. It is
    loaded from `checkpoints/dc-ae-f32c32-sana-1.1`.
- DINO (runs 2867528, 2863217) needs only torch and timm.
- Data: the preprocessed HDF5 files carry split/scene/frame metadata. No TruckScenes devkit
  or manifest is needed. Train: 83,554 samples / 431 scenes; val: 9,481 / 49.

## Target layout

```
c3pdar/
  data/truckscenes.py       Dataset + collate for the preprocessed HDF5 samples
  data/preprocessing/       reference copies of the HDF5 generation (not executed)
  models/
    dino.py                 DINOv3 backbone with per-modality LoRA
    dit.py                  RadarGen/SANA DiT, PPP path only (t=0, constant text buffer)
    dcae.py                 DC-AE condition encoder + full-width decoder
    heads.py                RoPE transformer neck, attention pixel-shuffle head
    builder.py              build_model(config) for "dino" | "dit"
  loss.py                   PPP likelihood + RCS/Doppler mark losses
  sampling.py               fields -> point clouds (budget, seed)
  module.py                 LightningModule + DataModule (if Lightning, see questions)
  evaluation/
    store.py                per-scene field HDF5 store (read / append)
    metrics.py              point-cloud and PPP metrics
    results.py              load metric CSVs into DataFrames, join runs
configs/                    one base config per model family (dino.yaml, dit.yaml)
scripts/                    train.py, predict.py, evaluate.py
backwards_comp/             checkpoint conversion + equivalence tests against the student code
docs/                       plans and arguments
```

## Run outputs

The config used is written next to the checkpoints; loading a model only needs the run directory.

```
outputs/<run>/
  config.yaml
  checkpoints/<step>.ckpt
  validation/<checkpoint>/
    fields/<scene_token>.h5     dense predicted fields, one group per frame (and per draw for stochastic models)
    points/<scene_token>.h5     optional drawn clouds per (budget, seed), e.g. for VoxelNeXt
    metrics.csv                 long format, see below
```

- `predict.py` (GPU) writes `fields/`. It opens files in append mode and skips frames that
  already exist. It can therefore first run on 50 scenes and later on the rest, or add
  further draws per input. One file per scene lets ranks write in parallel without merging.
- `evaluate.py` (CPU) is the only metric loop. It reads fields and ground truth, draws
  points per (budget, seed), and appends rows to `metrics.csv`.
- CSV columns: `run, checkpoint, scene_token, frame_idx, sample_token, draw, budget, seed,
  metric, value`. External results (VoxelNeXt detection) use the same columns, so they join
  on `sample_token`.

## Equivalence with the old code (`backwards_comp/`)

1. `convert_checkpoint.py`: original `.pth` -> new parameter names + `config.yaml`.
2. `dump_reference.py` runs in the original environment on fixed sample indices. It saves model
   outputs, losses, gradients and parameters after k optimizer steps, starting from a
   trained checkpoint.
3. `test_equivalence.py` loads the converted checkpoint and checks:
   - forward and loss match in fp32 (tight tolerance);
   - the k training steps match (tight in fp32, loose in bf16).
4. Metrics: re-evaluate one of the original SS50 validations and compare per-frame values.

## Order of work

1. Restructure the current port into the layout above.
2. Port the DiT/DC-AE path; conversion and load checks for all three checkpoints.
3. Training script and the equivalence tests.
4. Prediction store, evaluation loop, results loader.
