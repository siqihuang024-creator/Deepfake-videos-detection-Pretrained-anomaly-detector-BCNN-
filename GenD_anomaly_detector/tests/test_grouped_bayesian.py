"""Source-balanced fitting and covariance-preserving per-video posterior checks."""
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gad import bayesian_svdd as svdd


class GroupedBayesianTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.stack = ExitStack()
        for name, value in (("WIDTHS", (4, 2)), ("MAX_EPOCHS", 2), ("PATIENCE", 2),
                            ("BATCH_SIZE", 2), ("VALIDATION_DRAWS", 3), ("POSTERIOR_DRAWS", 6)):
            self.stack.enter_context(patch.object(svdd, name, value))
        self.rng = np.random.default_rng(123)

    def tearDown(self):
        self.stack.close()

    def test_each_source_has_equal_total_weight_despite_unequal_clip_counts(self):
        weights, count = svdd.source_balanced_weights(["a", "a", "a", "b"])
        np.testing.assert_allclose(weights, [1 / 6, 1 / 6, 1 / 6, .5])
        self.assertEqual(count, 2)
        self.assertAlmostEqual(float(weights[:3].sum()), float(weights[3]))
        for bad in ([], [""], [None], ["a", np.nan], [["a"]], [1, "a"]):
            with self.assertRaises(ValueError):
                svdd.source_balanced_weights(bad)

    def fit_head(self):
        train = np.array([[0., 0.], [0., 0.], [10., 2.]], np.float32)
        validation = np.array([[-1., 3.], [1., 1.]], np.float32)
        head = svdd.fit_source_balanced(train, validation, ["a", "a", "b"], ["v1", "v2"], "cpu")
        return train, validation, head

    def test_weighted_centering_group_count_and_finite_training_are_saved(self):
        train, validation, head = self.fit_head()
        np.testing.assert_allclose(head["feature_mean"], [5., 1.])
        np.testing.assert_allclose(head["training_weights"], [.25, .25, .5])
        self.assertEqual(head["effective_train_groups"], 2)
        self.assertEqual(head["early_stop_source_groups"], 2)
        self.assertTrue(np.isfinite(head["training_history"]).all())
        scores, extras = svdd.score(validation, head, "cpu")
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue((extras["posterior_distance_variance"] >= 0).all())
        self.assertEqual(len(head["training_history"]), 2)

    def test_source_overlap_and_incomplete_weight_recipes_are_refused(self):
        x = np.ones((3, 2), np.float32)
        v = np.ones((2, 2), np.float32)
        with self.assertRaisesRegex(ValueError, "crosses"):
            svdd.fit_source_balanced(x, v, ["a", "a", "b"], ["a", "v"], "cpu")
        with self.assertRaisesRegex(ValueError, "explicit"):
            svdd.fit(x, v, "cpu", training_weights=[1, 1, 1])
        for bad in ([1, 0, 1], [1, -1, 1], [1, np.nan, 1], [1, 1]):
            with self.assertRaises(ValueError):
                svdd.fit(x, v, "cpu", training_weights=bad, validation_weights=[1, 1], effective_train_groups=2)
        for count in (True, 0, 4, 2.5):
            with self.assertRaises(ValueError):
                svdd.fit(x, v, "cpu", training_weights=[1, 1, 1], validation_weights=[1, 1], effective_train_groups=count)

    def test_uniform_singleton_sources_match_the_legacy_fit(self):
        train = self.rng.normal(size=(6, 2)).astype(np.float32)
        validation = self.rng.normal(size=(3, 2)).astype(np.float32)
        with patch.object(svdd, "BATCH_SIZE", 99):
            old = svdd.fit(train, validation, "cpu")
            weighted = svdd.fit_source_balanced(train, validation, [f"t{i}" for i in range(6)],
                                                [f"v{i}" for i in range(3)], "cpu")
        for key in ("feature_mean", "center", "first_mean", "second_mean", "first_std", "second_std"):
            np.testing.assert_allclose(old[key], weighted[key], rtol=1e-5, atol=1e-6)
        self.assertNotIn("training_weights", old)

    def test_group_variance_includes_shared_draw_covariance(self):
        x, center = torch.tensor([[1.], [2.]]), torch.zeros(1)
        draws = iter([1., 2., 3.])
        def sample():
            return torch.tensor([[next(draws)]]), torch.ones(1, 1)
        grouping = (torch.tensor([0, 0]), torch.tensor([.5, .5]), 1)
        mean, variance = svdd.posterior_distances(x, center, sample, 3, grouping)
        per_draw_video = np.array([2.5, 10., 22.5])
        np.testing.assert_allclose(mean.numpy(), [per_draw_video.mean()], rtol=1e-6)
        np.testing.assert_allclose(variance.numpy(), [per_draw_video.var(ddof=1)], rtol=1e-6)
        incorrect_mean_of_clip_variances = np.mean([np.var([1., 4., 9.], ddof=1),
                                                    np.var([4., 16., 36.], ddof=1)])
        self.assertNotAlmostEqual(float(variance), incorrect_mean_of_clip_variances)

    def test_singleton_video_groups_match_individual_posterior_scores(self):
        _, validation, head = self.fit_head()
        individual, extras = svdd.score(validation, head, "cpu")
        names, grouped, grouped_extras = svdd.score_grouped(validation, head, ["b", "a"], "cpu")
        np.testing.assert_array_equal(names, ["a", "b"])
        np.testing.assert_allclose(grouped, individual[[1, 0]], rtol=1e-6)
        np.testing.assert_allclose(grouped_extras["posterior_distance_variance"],
                                   extras["posterior_distance_variance"][[1, 0]], rtol=1e-6)

    def test_positive_clip_weights_define_an_explicit_within_video_mean(self):
        head = {"center": np.zeros(1, np.float32), "first_mean": np.ones((1, 1), np.float32),
                "second_mean": np.ones((1, 1), np.float32), "first_std": np.zeros((1, 1), np.float32),
                "second_std": np.zeros((1, 1), np.float32)}
        names, scores, extras = svdd.score_grouped([[1.], [2.], [3.]], head, ["a", "a", "b"], "cpu", [1, 3, 4])
        np.testing.assert_array_equal(names, ["a", "b"])
        np.testing.assert_allclose(scores, [3.25, 9.])
        np.testing.assert_array_equal(extras["posterior_distance_variance"], [0, 0])
        for bad in ([1, 0, 1], [1, np.inf, 1], [1]):
            with self.assertRaises(ValueError):
                svdd.score_grouped([[1.], [2.], [3.]], head, ["a", "a", "b"], "cpu", bad)

    def test_grouped_scores_are_batch_invariant_and_survive_numeric_checkpoint_reload(self):
        _, _, head = self.fit_head()
        features = self.rng.normal(size=(9, 2)).astype(np.float32)
        groups = ["a"] * 3 + ["b"] * 4 + ["c"] * 2
        with patch.object(svdd, "BATCH_SIZE", 1):
            names, scores, extras = svdd.score_grouped(features, head, groups, "cpu")
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            path = Path(folder) / "head.npz"
            np.savez(path, **head)
            with np.load(path, allow_pickle=False) as arrays:
                loaded = {name: arrays[name] for name in arrays.files}
            with patch.object(svdd, "BATCH_SIZE", 99):
                names2, scores2, extras2 = svdd.score_grouped(features, loaded, groups, "cpu")
        np.testing.assert_array_equal(names, names2)
        np.testing.assert_allclose(scores, scores2, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(extras["posterior_distance_variance"], extras2["posterior_distance_variance"],
                                   rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
