"""CPU checks for the new sampling and posterior scoring rules."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import multivariate_t
from sklearn.neighbors import NearestNeighbors
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from build_identity_split import sample_training_fakes
from gad import anomaly, bayesian_svdd
from gad.manifest import OLD_FAKE, OLD_REAL


def test_method_quota():
    rows = [("train", OLD_REAL, "real", "r1"), ("train", OLD_REAL, "real", "r2")]
    rows.extend(("train", OLD_FAKE, method, f"{method}_{i}") for method in ("a", "b") for i in range(3))
    rows.append(("test", OLD_FAKE, "a", "test_a"))
    data = pd.DataFrame(rows, columns=["split", "label", "method", "path"])
    removed, _ = sample_training_fakes(data, ["a", "b"], 42, True, None)
    shuffled, _ = sample_training_fakes(data.sample(frac=1, random_state=0), ["a", "b"], 42, True, None)
    kept = data.drop(removed)
    assert removed.equals(shuffled)
    assert kept.query("split == 'train' and method != 'real'").groupby("method").size().to_dict() == {"a": 2, "b": 2}
    assert kept.loc[kept["split"].eq("test"), "path"].tolist() == ["test_a"]
    try:
        sample_training_fakes(data, ["a", "b", "missing"], 42, True, None)
    except ValueError as error:
        assert "missing=0" in str(error)
    else:
        raise AssertionError("Accepted a method without enough training videos")


def test_statistical_heads():
    rng = np.random.default_rng(1)
    train, val = rng.normal(size=(17, 24)), rng.normal(size=(10, 24))
    heads = anomaly.fit_statistical_heads(train, val)
    niw = heads["bayesian_gaussian"]
    chol = niw["psi_cholesky"]
    assert niw["nu0"] == 26 and niw["nu_n"] - 24 + 1 == 20
    score, draws = anomaly.niw_scores(val, niw)
    kappa = niw["kappa_n"]
    scale = chol @ chol.T * (kappa + 1) / (kappa * 20)
    np.testing.assert_allclose(score, -multivariate_t.logpdf(val, loc=niw["mean"], shape=scale, df=20))
    expected = niw["nu_n"] * (np.linalg.solve(chol, (val - niw["mean"]).T) ** 2).sum(axis=0) + 24 / kappa
    np.testing.assert_allclose(draws["posterior_distance_mean"], expected, rtol=0.15)
    assert (draws["posterior_distance_std"] > 0).all()
    actual, _ = anomaly.score_statistical_head("knn_cosine", val, heads["knn_cosine"])
    expected = NearestNeighbors(n_neighbors=5, metric="cosine").fit(train).kneighbors(val)[0][:, -1]
    np.testing.assert_allclose(actual, expected)
    for name in ("mahalanobis", "bayesian_gaussian_mixture"):
        score, _ = anomaly.score_statistical_head(name, val, heads[name])
        assert score.shape == (len(val),) and np.isfinite(score).all()


def test_svdd_posterior():
    torch.set_num_threads(2)
    rng = np.random.default_rng(3)
    train, val = rng.normal(loc=5, size=(48, 12)), rng.normal(loc=7, size=(10, 12))
    device = torch.device("cpu")
    head = bayesian_svdd.fit(train, val, device)
    np.testing.assert_allclose(head["feature_mean"], train.mean(axis=0), rtol=1e-6)
    assert np.abs(head["center"]).min() >= bayesian_svdd.CENTER_EPS
    assert 1 <= head["best_epoch"] <= head["epochs_run"]
    score, extras = bayesian_svdd.score(val, head, device)
    repeated, _ = bayesian_svdd.score(val, head, device)
    np.testing.assert_array_equal(score, repeated)
    assert (extras["posterior_distance_variance"] > 0).all()
    centered = torch.as_tensor(val, dtype=torch.float32) - torch.as_tensor(head["feature_mean"])
    direct, _ = bayesian_svdd.score_centered(centered, head)
    np.testing.assert_array_equal(score, direct.cpu().numpy())
    train_scores, _ = bayesian_svdd.score(train, head, device)
    np.testing.assert_allclose(head["train_score_variance"], np.var(train_scores), rtol=1e-6)
    assert not head["near_constant_train_scores"]
    assert not head["low_train_embedding_variance"]
    # Identical inputs expose both diagnostics even when posterior draws vary.
    collapsed = bayesian_svdd.training_diagnostics(torch.zeros(8, train.shape[1]), head, initial_variance=1.0)
    assert collapsed["near_constant_train_scores"] and collapsed["low_train_embedding_variance"]


if __name__ == "__main__":
    test_method_quota()
    test_statistical_heads()
    test_svdd_posterior()
    print("All sampling and anomaly checks passed")
