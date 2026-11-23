"""Train a C3P-dar model.

New run:  python scripts/train.py configs/dino_neck_compact.yaml --name my_run
Resume:   python scripts/train.py --resume outputs/my_run

A run directory RUNS_ROOT/<name> (default outputs/<name>) holds config.yaml, checkpoints/,
logs/metrics.csv and tensorboard/; existing runs are never overwritten.
"""

import argparse
from pathlib import Path

import lightning as L
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_only

from c3pdar import RUNS_ROOT
from c3pdar.module import C3PDarModule, C3PDarDataModule


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", type=Path)
    parser.add_argument("--name", help=f"run name under {RUNS_ROOT}")
    parser.add_argument("--resume", type=Path, help="run directory to continue from checkpoints/last.ckpt")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--num-nodes", type=int, default=1)
    args = parser.parse_args()

    if args.resume:
        run_dir = args.resume
        config = yaml.safe_load((run_dir / "config.yaml").read_text())
        checkpoint = run_dir / "checkpoints" / "last.ckpt"
    else:
        if args.config is None or args.name is None:
            parser.error("a new run needs CONFIG and --name")
        config = yaml.safe_load(args.config.read_text())
        run_dir = RUNS_ROOT / args.name
        checkpoint = None
        if rank_zero_only.rank == 0:
            run_dir.mkdir(parents=True, exist_ok=False)
            (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    train = config["train"]
    L.seed_everything(train["seed"], workers=True)
    module = C3PDarModule(config, pretrained=checkpoint is None)
    data = C3PDarDataModule(config["data"]["root"], train["batch_size"], config["data"].get("num_workers", 4),
                         seed=train["seed"])
    trainer = L.Trainer(
        accelerator="gpu", devices=args.devices, num_nodes=args.num_nodes, precision=train["precision"],
        max_steps=train["max_steps"], max_epochs=-1, gradient_clip_val=train["gradient_clip"],
        gradient_clip_algorithm="norm", use_distributed_sampler=False,
        log_every_n_steps=train["log_every_n_steps"],
        val_check_interval=train["val_every_n_steps"], check_val_every_n_epoch=None,
        limit_val_batches=train.get("limit_val_batches"), num_sanity_val_steps=0,
        logger=[CSVLogger(run_dir, name="", version="logs"),
                TensorBoardLogger(run_dir, name="", version="tensorboard", default_hp_metric=False)],
        callbacks=[ModelCheckpoint(dirpath=run_dir / "checkpoints", filename="step_{step}", auto_insert_metric_name=False,
                                   every_n_train_steps=train["checkpoint_every_n_steps"], save_top_k=-1,
                                   save_last=True)],
        default_root_dir=run_dir)
    trainer.fit(module, datamodule=data, ckpt_path=checkpoint)


if __name__ == "__main__":
    main()
