"""KL decomposition, original-trajectory parity and locked-data follow-up."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from gad import bayesian_svdd as original, bayesian_kl_ablation as ablation, segment_training
from run_segment_controls import run as run_controls
from run_segment_kl_ablation import run, verify_controls
import test_segment_training as fixtures


class BayesianKLAblationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.stack = ExitStack()
        recipe = dict(segment_training.RECIPE)
        for name, value in {"WIDTHS": (4, 2), "MAX_EPOCHS": 2, "PATIENCE": 2,
                             "BATCH_SIZE": 4, "VALIDATION_DRAWS": 3, "POSTERIOR_DRAWS": 4}.items():
            self.stack.enter_context(patch.object(original, name, value))
            recipe[name.lower()] = list(value) if name == "WIDTHS" else value
        self.stack.enter_context(patch.dict(segment_training.RECIPE, recipe))
        rng = np.random.default_rng(99)
        self.x, self.val = rng.normal(size=(6, 3)), rng.normal(size=(2, 3))
        self.groups, self.vg = ["a", "a", "b", "b", "c", "c"], ["d", "e"]

    def tearDown(self):
        self.stack.close()

    def test_same_beta_reproduces_original_parameters_and_losses(self):
        archived = original.fit_source_balanced(self.x, self.val, self.groups, self.vg, "cpu")
        detailed = ablation.fit(self.x, self.val, self.groups, self.vg, original.KL_WEIGHT)
        for key in ("first_mean", "second_mean", "first_std", "second_std", "center", "feature_mean",
                    "training_weights", "validation_weights", "best_epoch", "epochs_run"):
            np.testing.assert_allclose(detailed[key], archived[key], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(detailed["training_history"][:, 1], archived["training_history"][:, 1], rtol=1e-6)
        np.testing.assert_allclose(detailed["training_history"][:, 5], archived["training_history"][:, 2], rtol=1e-6)
        self.assertEqual(original.KL_WEIGHT, 1e-3)

    def test_loss_parts_and_beta_scaling_use_identical_batch_states(self):
        events = []
        with patch.object(original, "MAX_EPOCHS", 1), patch.object(original, "BATCH_SIZE", 100):
            high = ablation.fit(self.x, self.val, self.groups, self.vg, 1e-4, events.append)
            low = ablation.fit(self.x, self.val, self.groups, self.vg, 1e-5)
        h, l = high["training_history"], low["training_history"]
        np.testing.assert_allclose(h[:, 2], l[:, 2], rtol=1e-6)
        np.testing.assert_allclose(h[:, 3], l[:, 3], rtol=1e-6)
        np.testing.assert_allclose(h[:, 4], 10 * l[:, 4], rtol=1e-6)
        np.testing.assert_allclose(h[:, 1], h[:, 2] + h[:, 4], rtol=1e-6)
        np.testing.assert_allclose(h[:, 4], 1e-4 * h[:, 3] / 3, rtol=1e-6)
        self.assertAlmostEqual(events[0]["training_kl_per_source"], events[0]["training_kl_raw"] / 3)
        self.assertTrue(np.isfinite(high["first_median_abs_mean_over_std"]))

    def test_invalid_beta_and_overlapping_sources_rejected(self):
        for beta in (0, -1, True, float("nan"), float("inf"), "0.001"):
            with self.assertRaises(ValueError):
                ablation.fit(self.x, self.val, self.groups, self.vg, beta)
        with self.assertRaisesRegex(ValueError, "source crosses"):
            ablation.fit(self.x, self.val, self.groups, ["a", "e"], 1e-4)

    def test_completed_controls_required_and_all_four_fits_are_saved(self):
        fixture = fixtures.SegmentTrainingTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        inputs = fixture.make_inputs()
        reference, controls, out = (fixture.root / name for name in ("reference", "controls", "kl"))
        with redirect_stdout(io.StringIO()):
            segment_training.run(*inputs, reference)
        with self.assertRaises(FileNotFoundError):
            run(reference, controls, *inputs, out)
        self.assertFalse(out.exists())
        with redirect_stdout(io.StringIO()):
            run_controls(reference, *inputs, controls)
            run(reference, controls, *inputs, out)
        complete = json.loads((out / "completion.json").read_text())
        self.assertEqual(complete["kl_weights"], [1e-4, 1e-5])
        self.assertEqual(len(pd.read_csv(out / "comparison.csv")), 28)
        for encoder in ("official", "finetuned"):
            for variant, beta in ablation.VARIANTS.items():
                folder = out / encoder / variant
                with np.load(folder / "head.npz", allow_pickle=False) as saved:
                    self.assertEqual(float(saved["kl_weight"]), beta)
                    self.assertEqual(saved["first_mean"].shape, (4, 1024))
                    self.assertEqual(saved["first_std"].shape, (4, 1024))
                history = pd.read_csv(folder / "training_history.csv")
                np.testing.assert_allclose(history.training_loss,
                    history.training_distance_loss + history.training_weighted_kl, rtol=1e-6)
                lock = json.loads((folder / "kl_lock.json").read_text())
                self.assertEqual(lock["head_sha256"], segment_training.sha256_file(folder / "head.npz"))
        pd.testing.assert_frame_equal(pd.read_csv(reference / "training_units.csv", dtype=str),
                                      pd.read_csv(out / "training_units.csv", dtype=str))
        with self.assertRaises(FileExistsError):
            run(reference, controls, *inputs, out)
        comparison = controls / "comparison.csv"
        frame = pd.read_csv(comparison)
        frame.loc[0, "auroc"] += .1
        frame.to_csv(comparison, index=False)
        config = json.loads((reference / "training_registration.json").read_text())
        with self.assertRaisesRegex(ValueError, "comparison differs"):
            verify_controls(controls, reference, config)


if __name__ == "__main__":
    unittest.main()
