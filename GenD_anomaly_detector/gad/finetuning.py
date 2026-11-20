"""Full-encoder ablation outside the unmodified upstream checkout."""

import math

import torch
from torch import nn, optim


def grouped_parameters(model, lr, backbone_lr, weight_decay):
    if lr <= 0 or backbone_lr <= 0:
        raise ValueError("Learning rates must be positive")
    norm_parameters = {
        id(p) for module in model.feature_extractor.modules()
        if isinstance(module, nn.LayerNorm) for p in module.parameters(recurse=False)
    }
    buckets = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        fast = name.startswith("model.") or id(parameter) in norm_parameters
        role = "head_and_layernorm" if fast else "backbone"
        rate = lr if fast else backbone_lr
        no_decay = "bias" in name or "norm" in name
        decay = 0.0 if no_decay else weight_decay
        bucket = buckets.setdefault((role, no_decay), {"params": [], "names": [], "lr": rate,
                                                      "weight_decay": decay, "role": role})
        bucket["params"].append(parameter)
        bucket["names"].append(name)
    groups = list(buckets.values())
    actual = [id(p) for group in groups for p in group["params"]]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Optimizer groups must cover every trainable parameter exactly once")
    return groups


def group_report(groups, min_ratio):
    return [{"role": g["role"], "lr": g["lr"], "min_lr": g["lr"] * min_ratio,
             "weight_decay": g["weight_decay"], "parameters": sum(p.numel() for p in g["params"]),
             "tensors": len(g["params"]), "names": g["names"]} for g in groups]


def schedule_factor(step, batches, config):
    """Relative version of upstream's initial warmup then cyclic/cosine LR."""
    ratio = config.min_lr / config.lr
    warmup = int(config.warmup_epochs * batches)
    if warmup and step < warmup:
        return ratio + (1 - ratio) * step / warmup
    step = max(0, step - warmup)
    if config.lr_scheduler == "cyclic":
        period = int(config.num_epochs_in_cycle * batches)
        phase = (step % period) / period
    elif config.lr_scheduler == "cosine":
        phase = step / (config.max_epochs * batches)
    else:
        raise ValueError(f"Unsupported scheduler: {config.lr_scheduler}")
    return ratio + (1 - ratio) * (1 + math.cos(math.pi * phase)) / 2


def build_model(config, trainable="layernorm", backbone_lr=1e-5, gradient_checkpointing=False, verbose=True):
    from .upstream import add_gend_to_path
    add_gend_to_path()
    from src.model.GenD import GenD

    class FullGenD(GenD):
        def configure_optimizers(self):
            # Lightning can configure optimizers before attaching the loader.
            # Upstream explicitly sets up data here for its step-based schedule.
            self.trainer.fit_loop.setup_data()
            groups = grouped_parameters(self, config.lr, backbone_lr, config.weight_decay)
            self.optimizer_group_report = group_report(groups, config.min_lr / config.lr)
            optimizer = optim.AdamW(groups, betas=config.betas)
            batches = len(self.trainer.train_dataloader)
            scheduler = optim.lr_scheduler.LambdaLR(
                optimizer, lambda step: schedule_factor(step, batches, config)
            )
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}

    model = (FullGenD if trainable == "full" else GenD)(config, verbose=verbose)
    if trainable == "full":
        if not all(p.requires_grad for p in model.feature_extractor.parameters()):
            raise ValueError("Full fine-tuning requires every encoder parameter to be trainable")
    if gradient_checkpointing:
        setter = getattr(model.feature_extractor.backbone, "set_grad_checkpointing", None)
        if setter is None:
            raise ValueError("This backbone does not support gradient checkpointing")
        setter(True)
    return model
