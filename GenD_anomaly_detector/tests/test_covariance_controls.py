"""Weighted covariance, exact cross terms, frozen-reference replay and locks."""
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
from gad import covariance_controls as cov, feature_geometry as geometry, segment_training, bayesian_svdd as bayes
from gad.feature_cache import sha256_file, model_state_hash, signature_hash
import run_covariance_controls as followup
import run_geometry_diagnostics as geometry_run
from run_segment_controls import run as controls_run
from test_feature_geometry import FrozenModel
import test_segment_training as fixtures


class CovarianceControlTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_equal_source_covariance_contains_within_and_is_replication_invariant(self):
        x = np.array([[-1., 0.], [1., 2.], [4., -2.], [2., 3.]])
        groups = ["a", "a", "b", "c"]
        mean, c, info = cov.clip_covariance(x, groups)
        w = np.array([1/6, 1/6, 1/3, 1/3])
        np.testing.assert_allclose(mean, w @ x)
        np.testing.assert_allclose(c, np.cov(x, rowvar=False, aweights=w, ddof=0))
        self.assertGreater(info["within_trace"], 0)
        self.assertAlmostEqual(info["total_trace"], info["between_trace"]+info["within_trace"])
        repeated = cov.clip_covariance(np.vstack((x, x[:2])), groups+["a", "a"])
        np.testing.assert_allclose(repeated[0], mean)
        np.testing.assert_allclose(repeated[1], c)

    def test_regularized_distance_and_fixed_old_null_cross_terms_match_direct_inverse(self):
        c = np.array([[2., .8], [.8, 1.]])
        head = cov.regularize(np.array([.2, -.4]), c, .3)
        x = np.array([[1., 2.], [-1., 3.], [.2, -.4]])
        reference_null = np.array([[1.], [0.]])
        part = cov.distance_parts(x, head, reference_null)
        precision = np.linalg.inv(.7*c + .3*np.trace(c)/2*np.eye(2))
        delta = x-head["mean"]
        expected = np.einsum("ij,jk,ik->i", delta, precision, delta)
        np.testing.assert_allclose(part["score"], expected)
        np.testing.assert_allclose(part["score"], part["old_null_self"]+part["old_support_self"]+part["old_cross"])
        self.assertGreater(np.abs(part["old_cross"]).max(), .1)
        self.assertTrue((part["own_null_share"] == 0).all())
        with self.assertRaises(ValueError):
            cov.regularize(head["mean"], c, 0)

    def test_null_floor_decomposition_recovery_and_projection_covariance(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(6, 12))
        groups = ["a", "a", "b", "b", "c", "c"]
        old = geometry.fit_distances(x, groups)
        head = cov.recover_source_mean_head(old)
        self.assertEqual(head["empirical_null_basis"].shape, (12, 10))
        query = rng.normal(size=(7, 12))
        parts = cov.distance_parts(query, head, head["empirical_null_basis"])
        np.testing.assert_allclose(parts["old_cross"], 0, atol=1e-8)
        np.testing.assert_allclose(parts["own_null_contribution"], parts["old_null_self"], atol=1e-8)
        self.assertTrue((parts["own_null_share"] >= 0).all())
        self.assertTrue((parts["own_null_share"] <= 1+1e-8).all())
        u = rng.normal(size=12); u /= np.linalg.norm(u)
        _, full_c, _ = cov.clip_covariance(x, groups)
        _, reduced_c, _ = cov.clip_covariance(geometry.remove_direction(x, u), groups)
        transform = geometry.remove_direction(np.eye(12), u)
        np.testing.assert_allclose(reduced_c, transform.T @ full_c @ transform, atol=1e-12)

    def test_inspection_selection_uses_original_calibration_only_and_stays_bounded(self):
        frame = pd.DataFrame({"video_key": ["a", "b", "c"], "head_role": ["calibration", "dev", "early_stop"],
                              "label": [0, 1, 0], "centroid_l2": [2., 100., 300.]})
        target = followup.choose_inspection_targets(frame, 1.)
        self.assertEqual(target.video_key.tolist(), ["a"])
        with self.assertRaises(ValueError):
            followup.choose_inspection_targets(pd.concat([frame.iloc[:1]]*7), 1.)

    def test_full_pipeline_replays_342_emits_both_spaces_and_detects_changed_direction(self):
        fixture = fixtures.SegmentTrainingTests(); fixture.setUp()
        self.addCleanup(fixture.tearDown)
        inputs = fixture.make_inputs()
        _, cache_run, _ = inputs
        models = {"official": FrozenModel(7), "finetuned": FrozenModel(8)}
        for enc, model in models.items():
            receipt_path = cache_run / "cache_receipts" / (enc + ".json")
            receipt = json.loads(receipt_path.read_text())
            path = Path(receipt["cache"]) / "metadata.json"
            metadata = json.loads(path.read_text())
            metadata["signature"].update(weights_sha256=model_state_hash(model), head_configuration="LinearNorm",
                official_revision="pinned" if enc == "official" else None)
            metadata["signature_sha256"] = signature_hash(metadata["signature"])
            path.write_text(json.dumps(metadata), encoding="utf-8")
            receipt["metadata_sha256"] = sha256_file(path)
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        reference, controls, old, out = [fixture.root / n for n in ("reference", "controls", "old_geometry", "followup")]
        with ExitStack() as stack:
            recipe = dict(segment_training.RECIPE)
            for name, value in {"WIDTHS": (4, 2), "MAX_EPOCHS": 2, "PATIENCE": 2,
                    "BATCH_SIZE": 4, "VALIDATION_DRAWS": 3, "POSTERIOR_DRAWS": 4}.items():
                stack.enter_context(patch.object(bayes, name, value))
                recipe[name.lower()] = list(value) if name == "WIDTHS" else value
            stack.enter_context(patch.dict(segment_training.RECIPE, recipe))
            stack.enter_context(patch.dict(geometry_run.RECIPE, bootstrap_draws=10))
            stack.enter_context(patch.dict(followup.RECIPE, bootstrap_draws=10))
            def loader(model, device, revision):
                return models["official" if model == "official" else "finetuned"], None
            with redirect_stdout(io.StringIO()):
                segment_training.run(*inputs, reference)
                controls_run(reference, *inputs, controls)
                geometry_run.run(reference, controls, *inputs, old, loader=loader)
                followup.run(reference, controls, old, *inputs, out)
            completion = json.loads((out / "completion.json").read_text())
            self.assertTrue(completion["completed"])
            self.assertEqual(len(pd.read_csv(out / "comparison.csv")), 40)
            for name, digest in completion["artifacts_sha256"].items():
                self.assertEqual(sha256_file(out / name), digest)
            for enc in models:
                a = pd.read_csv(out / enc / "all_unit_scores.csv")
                b = pd.read_csv(old / enc / "all_unit_scores.csv")
                for name in followup.SCORE_NAMES:
                    np.testing.assert_allclose(a[name], b[name], rtol=1e-6, atol=1e-8)
                decomposition = pd.read_csv(out / enc / "distance_decomposition.csv")
                self.assertEqual(decomposition["head"].nunique(), 6)
                np.testing.assert_allclose(decomposition.score,
                    decomposition.old_null_self+decomposition.old_support_self+decomposition.old_cross, rtol=1e-6)
                with np.load(out / enc / "distance_heads.npz") as heads:
                    for space in ("original", "residual"):
                        for alpha in (30, 50):
                            name = "mahalanobis_clip_%s_a%d/shrinkage" % (space, alpha)
                            self.assertAlmostEqual(float(heads[name]), alpha/100.)
            with self.assertRaises(FileExistsError):
                followup.run(reference, controls, old, *inputs, out)
            directions = old / "classifier_directions.npz"
            directions.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Geometry reference"):
                followup.run(reference, controls, old, *inputs, fixture.root / "rejected")
            self.assertFalse((fixture.root / "rejected").exists())
