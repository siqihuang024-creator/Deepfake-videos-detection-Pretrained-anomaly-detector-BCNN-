"""Analytical weighting checks and reference-locked CPU control round trip."""
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
from gad import bayesian_svdd as bayes, segment_training as original, svdd_controls as controls
from run_segment_controls import run, load_reference
import test_segment_training as fixtures


class SVDDControlTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.stack = ExitStack()
        self.recipe = dict(original.RECIPE)
        for name, value in {"WIDTHS": (4, 2), "MAX_EPOCHS": 2, "PATIENCE": 2,
                             "BATCH_SIZE": 4, "VALIDATION_DRAWS": 3, "POSTERIOR_DRAWS": 4}.items():
            self.stack.enter_context(patch.object(bayes, name, value))
            self.recipe[name.lower()] = list(value) if name == "WIDTHS" else value
        self.stack.enter_context(patch.dict(original.RECIPE, self.recipe))

    def tearDown(self):
        self.stack.close()

    def test_centroid_is_source_balanced_and_rejects_bad_input(self):
        x = np.array([[0., 0.], [0., 0.], [10., 2.]])
        head = controls.fit_centroid(x, ["a", "a", "b"])
        np.testing.assert_allclose(head["feature_mean"], [5, 1])
        np.testing.assert_allclose(controls.score_centroid(x, head), [26, 26, 26])
        self.assertEqual(head["effective_train_groups"], 2)
        with self.assertRaises(ValueError):
            controls.fit_centroid(x, ["a"])
        with self.assertRaises(ValueError):
            controls.fit_centroid(np.full((3, 2), np.nan), ["a", "a", "b"])

    def test_point_gaussian_prior_matches_bayesian_mean_dependent_KL(self):
        first = torch.tensor([[1., 2.], [3., 4.]])
        second = torch.tensor([[2., 1.]])
        expected = .5 * (first.square().sum() / 1. + second.square().sum() / 1.)
        self.assertAlmostEqual(float(controls.point_prior(first, second)), float(expected))

    def test_initial_matrices_center_and_source_weights_match_Bayesian(self):
        rng = np.random.default_rng(5)
        x, val = rng.normal(size=(6, 3)), rng.normal(size=(2, 3))
        groups, vg = ["a", "a", "b", "b", "c", "c"], ["d", "e"]
        with patch.object(bayes, "LEARNING_RATE", 0.), patch.object(bayes, "MAX_EPOCHS", 1):
            point = controls.fit_deterministic(x, val, groups, vg)
            sampled = bayes.fit_source_balanced(x, val, groups, vg, "cpu")
        for key in ("first_mean", "second_mean", "center", "feature_mean", "training_weights", "validation_weights"):
            np.testing.assert_allclose(point[key], sampled[key], rtol=1e-6, atol=1e-7)
        self.assertNotIn("first_std", point)
        np.testing.assert_allclose(point["training_history"][:, 3], 0.)
        with self.assertRaisesRegex(ValueError, "source crosses"):
            controls.fit_deterministic(x, val, groups, ["a", "e"])

    def test_deterministic_histories_have_explicit_prior_and_round_trip(self):
        rng = np.random.default_rng(6)
        x, val = rng.normal(size=(6, 3)), rng.normal(size=(2, 3))
        groups, vg = ["a", "a", "b", "b", "c", "c"], ["d", "e"]
        plain = controls.fit_deterministic(x, val, groups, vg)
        prior = controls.fit_deterministic(x, val, groups, vg, prior_matched=True)
        np.testing.assert_allclose(plain["training_history"][:, 3], 0.)
        self.assertTrue((prior["training_history"][:, 3] > 0).all())
        h = prior["training_history"]
        np.testing.assert_allclose(h[:, 1], h[:, 2] + h[:, 3], rtol=1e-6)
        with io.BytesIO() as stream:
            np.savez(stream, **plain)
            stream.seek(0)
            with np.load(stream, allow_pickle=False) as saved:
                np.testing.assert_allclose(controls.score_deterministic(val, dict(saved)), controls.score_deterministic(val, plain))

    def test_controls_replay_reference_keep_units_and_refuse_changed_reference(self):
        fixture = fixtures.SegmentTrainingTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        inputs = fixture.make_inputs()
        reference = fixture.root / "reference"
        destination = fixture.root / "controls"
        with redirect_stdout(io.StringIO()):
            original.run(*inputs, reference)
            run(reference, *inputs, destination)
        completion = json.loads((destination / "completion.json").read_text())
        self.assertEqual(completion["phase"], "development_controls")
        self.assertTrue(completion["completed"])
        original_units = pd.read_csv(reference / "training_units.csv", dtype=str, keep_default_na=False)
        control_units = pd.read_csv(destination / "training_units.csv", dtype=str, keep_default_na=False)
        pd.testing.assert_frame_equal(original_units, control_units)
        summary = pd.read_csv(destination / "comparison.csv")
        self.assertEqual(len(summary), 20)
        self.assertEqual(summary["head"].nunique(), 5)
        for encoder in ("official", "finetuned"):
            lock = json.loads((destination / encoder / "control_lock.json").read_text())
            self.assertEqual(lock["heads_sha256"], original.sha256_file(destination / encoder / "control_heads.npz"))
            with np.load(destination / encoder / "control_heads.npz", allow_pickle=False) as saved:
                self.assertIn("centroid_l2/feature_mean", saved.files)
                self.assertIn("deterministic_svdd/first_mean", saved.files)
                self.assertNotIn("deterministic_svdd/first_std", saved.files)
        with self.assertRaises(FileExistsError):
            run(reference, *inputs, destination)
        changed = reference / "training_units.csv"
        changed.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unit table changed"):
            load_reference(reference, *inputs)


if __name__ == "__main__":
    unittest.main()
