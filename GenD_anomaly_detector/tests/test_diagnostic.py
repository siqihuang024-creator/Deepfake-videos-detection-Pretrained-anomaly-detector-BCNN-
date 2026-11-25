"""Leakage, cache-integrity and finite-metric regressions; no models or GPU needed."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from gad.metric_compat import calculate_eer
from gad.feature_cache import FrameCacheWriter, load_video_cache, sha256_file, frame_timing
from gad.diagnostic_protocol import audit_diagnostic_table, false_positive_groups, read_registered_plan, split_head_training
from prepare_diagnostic import prepare
from write_frame_lists import write_diagnostic_lists
from audit_official_overlap import audit


class MetricTests(unittest.TestCase):
    def test_perfect_prediction_has_finite_usable_threshold(self):
        p = np.array([.1, .2, .8, .9])
        eer, threshold = calculate_eer([0, 0, 1, 1], np.c_[1 - p, p], True)
        self.assertEqual(eer, 0.)
        np.testing.assert_array_equal(p > threshold, [False, False, True, True])

    def test_ties_and_reversed_predictions(self):
        for p, expected in ((np.full(4, .5), .5), (np.array([.9, .8, .2, .1]), 1.)):
            eer, threshold = calculate_eer([0, 0, 1, 1], np.c_[1 - p, p], True)
            self.assertAlmostEqual(eer, expected)
            self.assertTrue(np.isfinite(threshold))
        with self.assertRaises(ValueError):
            calculate_eer([0, 0], np.full((2, 2), .5))


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "tests")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_variable_frames_preserve_prequantization_video_mean_and_norms(self):
        paths = ["/c/d/real/a/frame_00001.png", "/c/d/real/b/frame_00003.png", "/c/d/real/a/frame_00004.png"]
        x = np.array([[.1234567, .9876543], [.5, .5], [.7777777, .2222222]], np.float32)
        dest = self.root / "cache"
        writer = FrameCacheWriter(dest, paths, {"weights": "A"})
        writer.write([2, 0], x[[2, 0]], [7., 8.], [.3, .1])
        with self.assertRaises(ValueError):
            writer.finish()
        writer.write([1], x[[1]], [9.], [.2])
        writer.finish()
        values = load_video_cache(dest, {"weights": "A"})
        np.testing.assert_allclose(values["features"][0], x[[0, 2]].astype(float).mean(0), rtol=1e-7)
        np.testing.assert_array_equal(values["num_frames"], [2, 1])
        np.testing.assert_array_equal(np.load(dest / "frame_norms.npy"), [8., 9., 7.])
        self.assertEqual(np.load(dest / "frame_l2.npy").dtype, np.dtype("float16"))
        self.assertTrue(np.isnan(np.load(dest / "frame_times_seconds.npy")).all())
        with self.assertRaises(ValueError):
            load_video_cache(dest, {"weights": "B"})
        with (dest / "video_features.npy").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaises(ValueError):
            load_video_cache(dest)

    def test_invalid_timing_does_not_leave_lock_or_partial_cache(self):
        with self.assertRaises(ValueError):
            FrameCacheWriter(self.root / "bad", ["/c/d/real/a/not_a_frame.png"], {})
        self.assertEqual(list(self.root.iterdir()), [])
        writer = FrameCacheWriter(self.root / "cache", ["/c/d/real/a/frame_00001.png"], {})
        writer.write([0], [[1., 0.]], [1.], [.2])
        with self.assertRaises(ValueError):
            writer.write([0], [[1., 0.]], [1.], [.2])
        writer.abort()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_recorded_pts_preserved_unknown_pts_not_invented(self):
        record = self.root / "d/real/a.json"
        record.parent.mkdir(parents=True)
        record.write_text(json.dumps({"selected": [1, 7], "selected_timestamps_seconds": [.04, None]}))
        indices, times = frame_timing(["/c/d/real/a/frame_00001.png", "/c/d/real/a/frame_00007.png"], self.root)
        np.testing.assert_array_equal(indices, [1, 7])
        self.assertEqual(times[0], .04)
        self.assertTrue(np.isnan(times[1]))


class ProtocolFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "tests")
        self.root = Path(self.temporary.name)
        self.protocol = self.root / "protocol"
        self.work = self.root / "preprocessed"
        rows = []
        for role, count in (("train", 8), ("calibration", 4), ("dev", 4), ("test", 4)):
            for i in range(count):
                label = int(role in ("dev", "test") and i >= 2)
                path = role + f"/{i}.mp4"
                (self.root / path).parent.mkdir(exist_ok=True)
                (self.root / path).touch()
                rows.append({"split": role, "dataset": "HDTF" if role in ("train", "calibration") else "DFDC",
                             "rel_path": path, "label": label, "source_clip": f"{role}_source{i % 2}" if role in ("dev", "test") else f"{role}_{i}",
                             "method": "swap" if label else "real", "target_id": f"{role}_person{i % 2}" if role in ("dev", "test") else f"{role}_person{i}",
                             "language": "en"})
        self.manifest = self.root / "manifest.csv"
        pd.DataFrame(rows).to_csv(self.manifest, index=False)
        self.selection = self.root / "selection.json"
        self.selection.write_text(json.dumps({"step": 2, "protocol": "celeb_identity", "ablation_run": "ln_b128_s42",
                                              "checkpoint": "/selected/model.ckpt", "checkpoint_sha256": "selected", "ablation_plan_sha256": "matrix"}))
        prepare(self.manifest, self.root, self.selection, self.protocol, "provided_ids", "Synthetic source/identity audit fixture")
        self.plan_path = self.protocol / "diagnostic_plan.json"
        self.plan, self.videos = read_registered_plan(self.plan_path)

    def tearDown(self):
        self.temporary.cleanup()

    def make_lists(self):
        from PIL import Image
        for row in self.videos.itertuples():
            dest = self.work / "crops" / row.video_key / "frame_00000.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 16), "red").save(dest)
            record = self.work / "records" / (row.video_key + ".json")
            record.parent.mkdir(parents=True, exist_ok=True)
            record.write_text(json.dumps({"split": row.split, "frame_subset": row.frame_subset,
                                          "status": "ok", "selected": [0]}))
        write_diagnostic_lists(self.plan_path, self.work, self.protocol / "lists", with_test=True)
        self.lists = self.protocol / "lists/frame_lists.json"

    def make_cache(self, test=False):
        list_path = self.protocol / "lists" / ("test/test.txt" if test else "development.txt")
        paths = list_path.read_text(encoding="utf-8").splitlines()
        signature = {"model": "official", "weights_sha256": "actualstate", "checkpoint_sha256": None,
                     "precision": "bf16", "gend_commit": "pinned", "preprocess": "fixed", "extractor_source_sha256": "source",
                     "custom_preprocessing": "None", "head_configuration": "normalized", "cache_algorithm": "v1",
                     "frame_list_sha256": sha256_file(list_path), "diagnostic_plan_sha256": sha256_file(self.plan_path)}
        directory = self.root / ("test_cache" if test else "development_cache")
        rng = np.random.default_rng(42 if not test else 43)
        x = rng.normal(size=(len(paths), 4)).astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        writer = FrameCacheWriter(directory, paths, signature)
        writer.write(np.arange(len(paths)), x, np.ones(len(paths)), rng.uniform(size=len(paths)))
        writer.finish()
        return directory


class ProtocolTests(ProtocolFixture):
    def test_head_early_stop_split_is_stable_and_keeps_connected_groups_together(self):
        rows = self.videos.loc[self.videos["split"].eq("train")].reset_index(drop=True)
        # Shared person links two different original videos.
        rows.loc[1, "target_id"] = rows.loc[0, "target_id"]
        # The same original links a third video transitively.
        rows.loc[2, "source_clip"] = rows.loc[1, "source_clip"]
        fit, val, info = split_head_training(rows)
        self.assertEqual(fit[0], fit[1])
        self.assertEqual(fit[1], fit[2])
        self.assertTrue((fit ^ val).all())
        _, _, shuffled = split_head_training(rows.sample(frac=1, random_state=3))
        self.assertEqual(info, shuffled)
        self.assertFalse(set(info["gradient_fit_video_keys"]) & set(info["early_stop_video_keys"]))
        invalid = rows.copy()
        invalid["split"] = "calibration"
        with self.assertRaises(ValueError):
            split_head_training(invalid)
        rows["target_id"] = "one_person"
        with self.assertRaises(ValueError):
            split_head_training(rows)

    def test_all_diagnostic_roles_uniform_without_changing_step2_sampling(self):
        from gad.manifest import frame_subset_for_split
        self.assertEqual(set(self.videos["frame_subset"]), {"uniform32"})
        self.assertEqual(frame_subset_for_split("train"), "random32")
        rows = self.videos.copy()
        rows.loc[rows["split"].eq("train"), "frame_subset"] = "random32"
        with self.assertRaises(ValueError):
            audit_diagnostic_table(rows, "provided_ids")

    def test_old_diagnostic_plan_is_not_silently_reused(self):
        plan = dict(self.plan)
        plan["version"] = 1
        self.plan_path.write_text(json.dumps(plan), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "register a new plan"):
            read_registered_plan(self.plan_path)

    def test_source_family_and_donor_leakage_rejected(self):
        for column in ("source_clip", "target_id", "donor_id"):
            rows = self.videos.copy()
            rows.loc[rows["split"].isin(("train", "dev")), column] = "crosses_roles"
            with self.assertRaises(ValueError):
                audit_diagnostic_table(rows, "provided_ids")
        rows = self.videos.copy()
        rows[["target_id", "donor_id", "driver_id"]] = ""
        result = audit_diagnostic_table(rows, "source_only")
        self.assertIn("NOT established", result["identity_claim"])

    def test_calibration_is_real_only_and_deepspeak_deferred(self):
        for column, mask, value in (("label", self.videos["split"].eq("calibration"), 1),
                                    ("dataset", self.videos["split"].eq("dev"), "DeepSpeak-v2")):
            rows = self.videos.copy()
            rows.loc[mask, column] = value
            with self.assertRaises(ValueError):
                audit_diagnostic_table(rows, "provided_ids")

    def test_lists_exclude_test_by_default_and_bind_plan(self):
        self.make_lists()
        registration = json.loads(self.lists.read_text(encoding="utf-8"))
        development = (self.protocol / "lists/development.txt").read_text(encoding="utf-8")
        for key in self.videos.loc[self.videos["split"].eq("test"), "video_key"]:
            self.assertNotIn(key, development)
        self.assertEqual(registration["plan_sha256"], sha256_file(self.plan_path))
        with self.assertRaises(FileExistsError):
            write_diagnostic_lists(self.plan_path, self.work, self.protocol / "lists")
        (self.protocol / "tables/videos.csv").write_text("changed")
        with self.assertRaises(ValueError):
            read_registered_plan(self.plan_path)

    def test_fpr_counts_by_language_and_source(self):
        rows = self.videos.loc[self.videos["split"].eq("dev")].reset_index(drop=True)
        result = false_positive_groups(rows, np.array([.2, .8, .7, .9]), .5, "provided_ids", 20)
        self.assertEqual(result[0]["false_positives"], 1)
        self.assertEqual(result[0]["observed_fpr"], .5)
        self.assertEqual(result[1]["language"], "en")
        source = false_positive_groups(rows, np.array([.2, .8, .7, .9]), .5, "source_only", 20)
        self.assertIn("actors appearing in multiple originals", source[0]["interval_limitation"])

    def test_official_partial_list_does_not_prove_no_overlap(self):
        official = self.root / "official.csv"
        candidate = self.root / "candidate.csv"
        pd.DataFrame({"dataset": ["CDFv3"], "canonical_video_id": ["one"]}).to_csv(official, index=False)
        pd.DataFrame({"dataset": ["CDFv3"], "canonical_video_id": ["two"], "split": ["test"]}).to_csv(candidate, index=False)
        result = audit(official, candidate, self.root / "overlap.json")
        self.assertEqual(result["status"], "unresolved_incomplete_official_list")


@unittest.skipUnless(importlib.util.find_spec("scipy") and importlib.util.find_spec("sklearn"), "SciPy/sklearn needed for CPU heads")
class EvaluationTests(ProtocolFixture):
    @unittest.skipUnless(importlib.util.find_spec("torch"), "Bayesian neural head needs Torch")
    def test_primary_bayesian_fits_only_train_and_parallel_can_test_from_same_cache(self):
        import torch
        torch.set_num_threads(2)
        from gad import anomaly, bayesian_svdd
        from evaluate_diagnostic import run
        self.make_lists()
        cache = self.make_cache()
        with patch.object(anomaly, "fit_statistical_heads", side_effect=AssertionError("Statistical heads are parallel")), patch.object(bayesian_svdd, "fit", wraps=bayesian_svdd.fit) as fitting, patch("builtins.print"):
            report = run(self.plan_path, self.lists, cache, self.root / "bayesian_report", profile="bayesian",
                         freeze_to=self.root / "bayesian_lock", examples=0, bootstrap_draws=20)
        training = report["head_training"]
        self.assertEqual(set(report["heads"]), {"bayesian_deep_svdd", "classifier"})
        self.assertEqual(len(training["gradient_fit_video_keys"]), 6)
        self.assertEqual(len(training["early_stop_video_keys"]), 2)
        self.assertEqual(report["calibration_videos"], 4)
        self.assertEqual(len(fitting.call_args[0][0]), 6)
        self.assertEqual(len(fitting.call_args[0][1]), 2)
        for key in training["gradient_fit_video_keys"] + training["early_stop_video_keys"]:
            self.assertEqual(self.videos.set_index("video_key").loc[key, "split"], "train")
        with np.load(self.root / "bayesian_lock/heads.npz", allow_pickle=False) as saved:
            cache_values = load_video_cache(cache)
            indices = {key: i for i, key in enumerate(cache_values["video_keys"])}
            expected_mean = cache_values["features"][[indices[key] for key in training["gradient_fit_video_keys"]]].mean(0)
            np.testing.assert_allclose(saved["bayesian_deep_svdd/feature_mean"], expected_mean, rtol=1e-5, atol=1e-7)
            self.assertEqual(saved["bayesian_deep_svdd/training_history"].shape[1], 4)
        self.assertTrue((self.root / "bayesian_report/training_progress.jsonl").is_file())
        self.assertEqual(sha256_file(self.root / "bayesian_report/heads.npz"), sha256_file(self.root / "bayesian_lock/heads.npz"))
        self.assertIn("posterior_distance_variance", (self.root / "bayesian_report/bayesian_deep_svdd_scores.csv").read_text(encoding="utf-8"))
        # Both planned configurations are frozen before opening the common final test.
        run(self.plan_path, self.lists, cache, self.root / "parallel_report", freeze_to=self.root / "parallel_lock",
            examples=0, bootstrap_draws=20)
        test_cache = self.make_cache(test=True)
        with patch.object(bayesian_svdd, "fit", side_effect=AssertionError("Must load locked head at test")):
            run(self.plan_path, self.lists, cache, self.root / "bayesian_test", phase="test", test_cache=test_cache,
                lock=self.root / "bayesian_lock", profile="bayesian", examples=0, bootstrap_draws=20)
        run(self.plan_path, self.lists, cache, self.root / "parallel_test", phase="test", test_cache=test_cache,
            lock=self.root / "parallel_lock", examples=0, bootstrap_draws=20)
        self.assertEqual(len(list((self.protocol / "final_test_receipts").glob("*.json"))), 2)
        with self.assertRaises(FileExistsError):
            run(self.plan_path, self.lists, cache, self.root / "bayesian_test_again", phase="test", test_cache=test_cache,
                lock=self.root / "bayesian_lock", profile="bayesian", examples=0, bootstrap_draws=20)
        with self.assertRaisesRegex(ValueError, "before any final testing"):
            run(self.plan_path, self.lists, cache, self.root / "late_lock_report", freeze_to=self.root / "late_lock",
                examples=0, bootstrap_draws=20)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Legacy Bayesian neural head check needs Torch")
    def test_existing_bayesian_network_posterior_regression(self):
        from test_anomaly import test_svdd_posterior
        test_svdd_posterior()

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Torch needed for frozen extraction smoke")
    def test_frozen_extraction_one_pass_and_cache_reuse(self):
        import torch
        from types import SimpleNamespace
        from PIL import Image
        import extract_frame_cache as extraction
        self.make_lists()

        class Encoder(torch.nn.Module):
            _preprocess = "toy-fixed-rgb"

            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.ones(1))
                self.calls = 0

            def forward(self, image):
                self.calls += 1
                return image.mean(dim=(2, 3)) * self.scale

        model = torch.nn.Module()
        model.feature_extractor = Encoder()
        model.model = torch.nn.Linear(3, 2)
        model.config = SimpleNamespace(head="LinearNorm", custom_preprocessing=None)

        def preprocess(image):
            return torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255

        receipt = self.root / "receipt.json"
        command = ["extract_frame_cache.py", "--model", "official", "--frame-list",
                   str(self.protocol / "lists/development.txt"), "--list-registration", str(self.lists),
                   "--cache-root", str(self.root / "cache_root"), "--receipt", str(receipt),
                   "--device", "cpu", "--precision", "fp32", "--num-workers", "0", "--batch-size", "99"]
        with patch.object(sys, "argv", command), patch.object(extraction, "load_frozen_module", return_value=(model, preprocess)), patch.object(extraction, "upstream_commit", return_value="toy"):
            extraction.main()
            extraction.main()
        self.assertEqual(model.feature_extractor.calls, 1)
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))
        cached = json.loads(receipt.read_text(encoding="utf-8"))["cache"]
        values = load_video_cache(cached)
        np.testing.assert_allclose(values["features"], np.tile([1., 0., 0.], (16, 1)))

    def test_score_cache_rejects_wrong_model_and_test_frames_in_development(self):
        from evaluate_diagnostic import registered_cache
        self.make_lists()
        cache = self.make_cache()
        meta_path = cache / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        from gad.feature_cache import signature_hash
        meta["signature"].update(model="other.ckpt", checkpoint_sha256="wrong")
        meta["signature_sha256"] = signature_hash(meta["signature"])
        meta_path.write_text(json.dumps(meta))
        with self.assertRaises(ValueError):
            registered_cache(cache, self.lists, self.plan_path, self.plan, self.videos)
        test_cache = self.make_cache(test=True)
        with self.assertRaises(ValueError):
            registered_cache(test_cache, self.lists, self.plan_path, self.plan, self.videos)

    def test_only_two_heads_calibrate_independently_freeze_and_evaluate_once(self):
        from gad import anomaly
        from evaluate_diagnostic import run
        self.make_lists()
        cache = self.make_cache()
        with patch.object(anomaly, "fit_mixture", side_effect=AssertionError("Must not fit BGMM")):
            report = run(self.plan_path, self.lists, cache, self.root / "dev_report", freeze_to=self.root / "lock", examples=2, bootstrap_draws=20)
        self.assertEqual(set(report["heads"]), {"mahalanobis", "bayesian_gaussian", "classifier"})
        self.assertEqual(report["calibration_videos"], 4)
        self.assertEqual(report["heads"]["mahalanobis"]["0.05"]["per_dataset"]["DFDC"]["num_real"], 2)
        heads = anomaly.fit_statistical_heads(np.random.default_rng(1).normal(size=(8, 4)), head_names=("bayesian_gaussian",))
        _, extras = anomaly.score_statistical_head("bayesian_gaussian", np.ones((3, 4)), heads["bayesian_gaussian"], posterior_draws=0)
        self.assertEqual(extras, {})
        # Development labels cannot alter either fitted heads or calibration thresholds.
        frozen = json.loads((self.root / "lock/method_lock.json").read_text(encoding="utf-8"))
        values = load_video_cache(cache)
        index = {key: i for i, key in enumerate(values["video_keys"])}
        cal_indices = [index[k] for k in self.videos.loc[self.videos["split"].eq("calibration"), "video_key"]]
        expected = np.quantile(values["fake_prob"][cal_indices], .95)
        self.assertAlmostEqual(frozen["thresholds"]["classifier"]["0.05"], expected)
        test_cache = self.make_cache(test=True)
        result = run(self.plan_path, self.lists, cache, self.root / "test_report", phase="test", test_cache=test_cache,
                     lock=self.root / "lock", bootstrap_draws=20)
        self.assertEqual(result["phase"], "test")
        with self.assertRaises(FileExistsError):
            run(self.plan_path, self.lists, cache, self.root / "test_again", phase="test", test_cache=test_cache,
                lock=self.root / "lock", bootstrap_draws=20)

    def test_test_without_lock_or_with_changed_recipe_refused(self):
        from evaluate_diagnostic import run
        self.make_lists()
        cache = self.make_cache()
        with self.assertRaises(ValueError):
            run(self.plan_path, self.lists, cache, self.root / "invalid", phase="test")
        run(self.plan_path, self.lists, cache, self.root / "development_report", freeze_to=self.root / "lock", examples=0, bootstrap_draws=20)
        test_cache = self.make_cache(test=True)
        with self.assertRaises(ValueError):
            run(self.plan_path, self.lists, cache, self.root / "changed", phase="test", test_cache=test_cache,
                lock=self.root / "lock", posterior_draws=2, bootstrap_draws=20)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Legacy anomaly check imports Torch")
    def test_legacy_statistical_heads_still_match_reference_distributions(self):
        from test_anomaly import test_statistical_heads
        test_statistical_heads()


if __name__ == "__main__":
    unittest.main()
