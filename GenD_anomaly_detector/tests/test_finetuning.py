"""CPU optimizer/scheduler checks with a tiny model; no model downloads."""

import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn, optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gad.finetuning import build_model, grouped_parameters, schedule_factor


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4), nn.Linear(4, 4))
        self.model = nn.Linear(4, 2)


class FinetuningChecks(unittest.TestCase):
    def test_full_optimizer_initializes_the_loader_before_sizing_its_schedule(self):
        class FakeGenD(Toy):
            def __init__(self, config, verbose=False):
                super().__init__()
                self.trainer = SimpleNamespace(train_dataloader=None)
                def setup_data():
                    self.trainer.train_dataloader = list(range(5))
                self.trainer.fit_loop = SimpleNamespace(setup_data=setup_data)

        upstream = ModuleType("src.model.GenD")
        upstream.GenD = FakeGenD
        cfg = SimpleNamespace(lr=3e-4, min_lr=1e-5, weight_decay=0,
                              betas=(0.9, 0.999), warmup_epochs=1,
                              lr_scheduler="cyclic", num_epochs_in_cycle=2, max_epochs=6)
        with patch.dict(sys.modules, {"src": ModuleType("src"), "src.model": ModuleType("src.model"),
                                      "src.model.GenD": upstream}):
            model = build_model(cfg, trainable="full", verbose=False)
            configured = model.configure_optimizers()
        self.assertEqual(len(model.trainer.train_dataloader), 5)
        self.assertEqual(configured["lr_scheduler"]["interval"], "step")
        rates = sorted({group["lr"] for group in configured["optimizer"].param_groups})
        self.assertEqual(len(rates), 2)
        self.assertAlmostEqual(rates[0], 1e-5 / 30, places=15)
        self.assertAlmostEqual(rates[1], 1e-5, places=15)

    def test_full_optimizer_covers_parameters_once_at_the_right_learning_rates(self):
        model = Toy()
        groups = grouped_parameters(model, 3e-4, 1e-5, 0.1)
        ids = [id(p) for group in groups for p in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters()})
        self.assertEqual({group["lr"] for group in groups}, {3e-4, 1e-5})
        high = {id(p) for g in groups if g["lr"] == 3e-4 for p in g["params"]}
        expected = {id(p) for p in model.model.parameters()} | {id(p) for p in model.feature_extractor[1].parameters()}
        self.assertEqual(high, expected)
        for group in groups:
            for name in group["names"]:
                self.assertEqual(group["weight_decay"], 0.0 if "bias" in name or "norm" in name else 0.1)
        optimizer = optim.AdamW(groups, betas=(0.9, 0.999))
        before = [p.detach().clone() for p in model.parameters()]
        optimizer.zero_grad()
        model.model(model.feature_extractor(torch.randn(5, 3))).square().mean().backward()
        optimizer.step()
        self.assertTrue(all(not torch.equal(a, b) for a, b in zip(before, model.parameters())))

    def test_relative_schedule_matches_upstream_for_each_supported_schedule(self):
        for kind in ("cyclic", "cosine"):
            cfg = SimpleNamespace(lr=3e-4, min_lr=1e-5, warmup_epochs=1,
                                  lr_scheduler=kind, num_epochs_in_cycle=2, max_epochs=6)
            batches = 5
            parameter = nn.Parameter(torch.zeros(1))
            optimizer = optim.AdamW([parameter], lr=cfg.lr)
            if kind == "cyclic":
                cosine = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=cfg.min_lr)
            else:
                cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=cfg.min_lr)
            warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=cfg.min_lr / cfg.lr, total_iters=5)
            scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[5])
            for step in range(31):
                self.assertAlmostEqual(optimizer.param_groups[0]["lr"],
                                       cfg.lr * schedule_factor(step, batches, cfg), places=12)
                optimizer.step()
                scheduler.step()


if __name__ == "__main__":
    unittest.main()
