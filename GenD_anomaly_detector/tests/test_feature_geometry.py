"""Source balancing, orthogonal geometry, frozen-state binding and full CPU path."""
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
from sklearn.covariance import LedoitWolf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from gad import feature_geometry as geometry, classifier_direction as directions
from gad import bayesian_svdd as bayes, segment_training
from gad.feature_cache import model_state_hash, sha256_file
from run_segment_controls import run as run_controls
from run_geometry_diagnostics import run, RECIPE, wilson
import test_segment_training as fixtures


class FrozenModel(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.feature_extractor = torch.nn.Identity()
        self.model = torch.nn.Module()
        self.model.normalize_inputs = True
        self.model.linear = torch.nn.Linear(1024, 2)
        with torch.no_grad():
            generator = torch.Generator().manual_seed(seed)
            self.model.linear.weight.copy_(torch.randn((2, 1024), generator=generator))
            self.model.linear.bias.zero_()


class FeatureGeometryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_source_means_covariance_match_sklearn_and_ignore_clip_replication(self):
        x = np.array([[0., 0.], [0., 0.], [3., 2.], [-2., 4.], [4., -3.]])
        groups = ["a", "a", "b", "c", "d"]
        names, means = geometry.source_means(x, groups)
        self.assertEqual(list(names), ["a", "b", "c", "d"])
        head = geometry.fit_distances(x, groups)
        expected = LedoitWolf(store_precision=False).fit(means)
        np.testing.assert_allclose(head["mean"], expected.location_)
        np.testing.assert_allclose(head["cholesky"] @ head["cholesky"].T, expected.covariance_, atol=1e-12)
        repeated = geometry.fit_distances(np.vstack((x, x[:2])), groups + ["a", "a"])
        np.testing.assert_allclose(head["mean"], repeated["mean"])
        np.testing.assert_allclose(head["cholesky"], repeated["cholesky"])
        l2, mahal = geometry.score_distances(x, head)
        delta = x - head["mean"]
        np.testing.assert_allclose(l2, (delta * delta).sum(1))
        np.testing.assert_allclose(mahal, np.einsum("ij,jk,ik->i", delta, np.linalg.inv(expected.covariance_), delta))

    def test_projection_removes_only_classifier_axis_without_renormalizing(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(9, 5))
        u = geometry.classifier_direction(np.vstack((np.zeros(5), np.arange(1, 6))))
        residual = geometry.remove_direction(x, u)
        self.assertEqual(residual.shape, (9, 4))
        np.testing.assert_allclose((x*x).sum(1), (x@u)**2 + (residual*residual).sum(1), atol=1e-12)
        np.testing.assert_allclose(residual, geometry.remove_direction(x + 9*u, u), atol=1e-12)
        projection = x - (x@u)[:, None]*u
        np.testing.assert_allclose(geometry.squared_pairs(residual, residual),
                                   geometry.squared_pairs(projection, projection), atol=1e-12)
        with self.assertRaises(ValueError):
            geometry.remove_direction(x, u*2)
        with self.assertRaises(ValueError):
            geometry.classifier_direction(np.zeros((2, 5)))

    def test_invalid_covariance_inputs_fail_without_silent_parameter_changes(self):
        with self.assertRaisesRegex(ValueError, "varying source means"):
            geometry.fit_distances(np.ones((4, 2)), ["a", "b", "c", "d"])
        with self.assertRaisesRegex(ValueError, "positive definite"):
            geometry.fit_distances(np.array([[-1., 0.], [1., 0.]]), ["a", "b"])
        with self.assertRaisesRegex(ValueError, "source ID"):
            geometry.source_means(np.ones((2, 2)), ["a", ""])

    def test_frozen_direction_binds_model_hash_revision_and_normalization(self):
        model = FrozenModel(5)
        signature = {"model": "official", "official_revision": "pinned", "weights_sha256": model_state_hash(model),
                     "head_configuration": "LinearNorm"}
        calls = []

        def loader(*args):
            calls.append(args)
            return model, None

        values, info = directions.read_direction(signature, loader)
        self.assertEqual(calls, [("official", "cpu", "pinned")])
        self.assertEqual(values["linear_weights"].shape, (2, 1024))
        np.testing.assert_allclose(values["direction"], geometry.classifier_direction(values["linear_weights"]))
        self.assertEqual(info["weights_sha256"], signature["weights_sha256"])
        with self.assertRaisesRegex(ValueError, "state differs"):
            directions.read_direction(dict(signature, weights_sha256="changed"), loader)
        model.model.normalize_inputs = False
        with self.assertRaisesRegex(ValueError, "L2-input"):
            directions.read_direction(signature, loader)

    def test_shift_kernel_uses_fitting_real_sources_only_and_ignores_fake_features(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(11, 3))
        rows = pd.DataFrame({"head_role": ["gradient_fit"]*4 + ["calibration"]*3 + ["dev"]*4,
                             "label": [0]*9 + [1]*2, "source_clip": [str(i) for i in range(11)]})
        first = geometry.real_feature_shift(x, rows)
        changed = x.copy()
        changed[-2:] += 1000
        self.assertEqual(first, geometry.real_feature_shift(changed, rows))
        changed[7:9] += 50
        shifted = geometry.real_feature_shift(changed, rows)
        self.assertEqual(first["kernel_gamma_from_fitting_sources_only"], shifted["kernel_gamma_from_fitting_sources_only"])
        self.assertGreater(shifted["comparisons"][1]["rbf_mmd_squared_biased"], first["comparisons"][1]["rbf_mmd_squared_biased"])
        self.assertGreater(wilson(0, 100)["high"], 0)

    def test_full_locked_geometry_pipeline_and_frozen_model_mismatch(self):
        fixture = fixtures.SegmentTrainingTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        inputs = fixture.make_inputs()
        data, cache_run, selection = inputs
        models = {"official": FrozenModel(7), "finetuned": FrozenModel(8)}
        for encoder, model in models.items():
            receipt_path = cache_run / "cache_receipts" / (encoder + ".json")
            receipt = json.loads(receipt_path.read_text())
            metadata_path = Path(receipt["cache"]) / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            signature = metadata["signature"]
            signature.update(weights_sha256=model_state_hash(model), head_configuration="LinearNorm",
                             official_revision="pinned" if encoder == "official" else None)
            from gad.feature_cache import signature_hash
            metadata["signature_sha256"] = signature_hash(signature)
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            receipt["metadata_sha256"] = sha256_file(metadata_path)
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        reference, controls, out = (fixture.root / name for name in ("reference", "controls", "geometry"))
        with ExitStack() as stack:
            recipe = dict(segment_training.RECIPE)
            for name, value in {"WIDTHS": (4, 2), "MAX_EPOCHS": 2, "PATIENCE": 2,
                               "BATCH_SIZE": 4, "VALIDATION_DRAWS": 3, "POSTERIOR_DRAWS": 4}.items():
                stack.enter_context(patch.object(bayes, name, value))
                recipe[name.lower()] = list(value) if name == "WIDTHS" else value
            stack.enter_context(patch.dict(segment_training.RECIPE, recipe))
            stack.enter_context(patch.dict(RECIPE, bootstrap_draws=10))

            def loader(model, device, revision):
                self.assertEqual(device, "cpu")
                return models["official" if model == "official" else "finetuned"], None

            with redirect_stdout(io.StringIO()):
                segment_training.run(*inputs, reference)
                run_controls(reference, *inputs, controls)
                run(reference, controls, *inputs, out, loader=loader)
            self.assertEqual(json.loads((out / "completion.json").read_text())["phase"], "development_geometry")
            self.assertEqual(len(pd.read_csv(out / "comparison.csv")), 24)
            for encoder in ("official", "finetuned"):
                folder = out / encoder
                lock = json.loads((folder / "geometry_lock.json").read_text())
                self.assertEqual(lock["heads_sha256"], sha256_file(folder / "distance_heads.npz"))
                with np.load(folder / "distance_heads.npz", allow_pickle=False) as saved:
                    self.assertEqual(saved["original/mean"].shape, (1024,))
                    self.assertEqual(saved["residual/mean"].shape, (1023,))
                    self.assertEqual(int(saved["original/source_count"]), 4)
                frame = pd.read_csv(folder / "all_unit_scores.csv")
                np.testing.assert_allclose(frame.centroid_l2,
                    frame.classifier_direction_only_l2 + frame.centroid_without_classifier_direction, rtol=1e-6, atol=1e-9)
                self.assertTrue((folder / "real_source_shift.json").is_file())
                self.assertTrue((folder / "real_source_score_summary.csv").is_file())
            with self.assertRaises(FileExistsError):
                run(reference, controls, *inputs, out, loader=loader)
