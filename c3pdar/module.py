"""The model and data module for C3P-dar training via pytorch lightning, with its optimizers and warmup schedule."""

import lightning as L
import torch
from torch.utils.data import DataLoader, DistributedSampler

from .data import PreprocessedRadarGenDataset, RadarGenBatch
from .loss import PPPLoss, weighted_loss
from .models import build_model


class C3PDarModule(L.LightningModule):
    def __init__(self, config, pretrained=True):
        super().__init__()
        self.save_hyperparameters({"config": config})
        self.config = config
        self.model = build_model(config["model"], pretrained=pretrained)
        loss_cfg = dict(config["loss"])
        self.weights = loss_cfg.pop("weights")
        self.loss = PPPLoss(**loss_cfg)

    def forward(self, batch):
        return self.model(*batch.normalized_conditions().unbind(1))

    def _losses(self, batch):
        fields = self(batch)
        losses = self.loss(fields, batch)
        return weighted_loss(losses, self.weights)

    def training_step(self, batch, batch_index):
        total, logs = self._losses(batch)
        self.log_dict({f"train/{k}": v for k, v in logs.items()}, batch_size=len(batch), prog_bar=False)
        self.log("train/lr", self.optimizers().param_groups[0]["lr"], batch_size=len(batch))
        return total

    def validation_step(self, batch, batch_index):
        _, logs = self._losses(batch)
        self.log_dict({f"val/{k}": v for k, v in logs.items()}, batch_size=len(batch), sync_dist=True)

    def configure_optimizers(self):
        optimizer = build_optimizer([p for p in self.model.parameters() if p.requires_grad], self.config["optim"])
        scheduler = warmup_schedule(optimizer, self.config["train"].get("warmup_steps", 0))
        return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval="step"))


class C3PDarDataModule(L.LightningDataModule):
    """Train split: all frames, a global seeded shuffle per epoch. Validation: val-split keyframes."""

    def __init__(self, root, batch_size, num_workers=4, seed=42, val_batch_size=None):
        super().__init__()
        self.root, self.batch_size, self.num_workers, self.seed = root, batch_size, num_workers, seed
        self.val_batch_size = val_batch_size or batch_size

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_set = PreprocessedRadarGenDataset(self.root, "train")
        if stage in (None, "fit", "validate"):
            self.val_set = PreprocessedRadarGenDataset(self.root, "val", keyframes_only=True)

    def _loader(self, dataset, batch_size, shuffle):
        # Explicit sampler: identical global order for any GPU count (Lightning calls set_epoch).
        sampler = DistributedSampler(dataset, num_replicas=self.trainer.world_size, rank=self.trainer.global_rank,
                                     shuffle=shuffle, seed=self.seed, drop_last=False)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=self.num_workers,
                          pin_memory=True, drop_last=False, collate_fn=RadarGenBatch.collate,
                          persistent_workers=self.num_workers > 0)

    def train_dataloader(self):
        return self._loader(self.train_set, self.batch_size, True)

    def val_dataloader(self):
        return self._loader(self.val_set, self.val_batch_size, False)


class CAME(torch.optim.Optimizer):
    """CAME (confidence-guided adaptive memory-efficient optimization) as used by SANA.

    Linear and 1x1-convolution weights use factored second moments; all other
    parameters use full (Adam-like) second moments.
    """

    def __init__(self, params, lr, eps=(1e-30, 1e-16), clip_threshold=1.0, betas=(0.9, 0.999, 0.9999),
                 weight_decay=0.0):
        super().__init__(params, dict(lr=lr, eps=eps, clip_threshold=clip_threshold, betas=betas,
                                      weight_decay=weight_decay))

    @staticmethod
    def _factored(shape):
        return len(shape) == 2 or (len(shape) == 4 and shape[2] == shape[3] == 1)

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / tensor.numel() ** 0.5

    @staticmethod
    def _approx_sq(row, col):
        return (row / row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1) * col.unsqueeze(-2).rsqrt()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2, beta3 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.float() if p.grad.dtype in (torch.float16, torch.bfloat16) else p.grad
                shape = grad.shape
                factored = self._factored(shape)
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(grad)
                    if factored:
                        for name, size in (("sq_row", shape[0]), ("sq_col", shape[1]),
                                           ("res_row", shape[0]), ("res_col", shape[1])):
                            state[f"exp_avg_{name}"] = torch.zeros(size).type_as(grad)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)
                state["step"] += 1

                update = grad ** 2 + group["eps"][0]
                if factored:
                    flat = update.reshape(shape[0], shape[1])
                    state["exp_avg_sq_row"].mul_(beta2).add_(flat.mean(dim=1), alpha=1 - beta2)
                    state["exp_avg_sq_col"].mul_(beta2).add_(flat.mean(dim=0), alpha=1 - beta2)
                    update = self._approx_sq(state["exp_avg_sq_row"], state["exp_avg_sq_col"]).view(shape)
                    update.mul_(grad)
                else:
                    state["exp_avg_sq"].mul_(beta2).add_(update, alpha=1 - beta2)
                    update = state["exp_avg_sq"].rsqrt().mul_(grad)
                update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))

                exp_avg = state["exp_avg"]
                exp_avg.mul_(beta1).add_(update, alpha=1 - beta1)
                if factored:
                    # Confidence: down-weight directions where the update disagrees with its average.
                    res = ((update - exp_avg) ** 2 + group["eps"][1]).reshape(shape[0], shape[1])
                    state["exp_avg_res_row"].mul_(beta3).add_(res.mean(dim=1), alpha=1 - beta3)
                    state["exp_avg_res_col"].mul_(beta3).add_(res.mean(dim=0), alpha=1 - beta3)
                    update = self._approx_sq(state["exp_avg_res_row"], state["exp_avg_res_col"]).view(shape)
                    update.mul_(exp_avg)
                else:
                    update = exp_avg.clone()

                if group["weight_decay"] != 0:
                    p.add_(p, alpha=-group["weight_decay"] * group["lr"])
                p.add_(update.mul_(group["lr"]).neg_())
        return loss


def build_optimizer(parameters, cfg):
    cfg = dict(cfg)
    kind = cfg.pop("type")
    if kind == "CAME":
        return CAME(parameters, **cfg)
    return getattr(torch.optim, kind)(parameters, **cfg)


def warmup_schedule(optimizer, warmup_steps):
    """Linear warmup from 0 (first update) to the base learning rate at ``warmup_steps``, then constant."""
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, step / warmup_steps)
                                             if warmup_steps else 1.0)
