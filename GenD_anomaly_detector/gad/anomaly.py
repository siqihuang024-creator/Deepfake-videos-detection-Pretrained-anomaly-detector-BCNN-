"""Real-only anomaly scores on frozen video embeddings."""

import math

import numpy as np
from scipy.linalg import solve_triangular
from scipy.linalg.blas import dtrmm
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture

BAYESIAN_PRIOR_MEAN_WEIGHT = 0.01
NIW_DRAWS = 100
KNN_NEIGHBORS = 5
MIXTURE_PCA_DIM = 64
MIXTURE_MAX_COMPONENTS = 10
SEED = 42


def matrix(features):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty, finite feature matrix.")
    return x


STATISTICAL_HEADS = ("mahalanobis", "knn_cosine", "bayesian_gaussian", "bayesian_gaussian_mixture")


def fit_statistical_heads(training, validation=None, head_names=None):
    x = matrix(training)
    names = tuple(STATISTICAL_HEADS if head_names is None else head_names)
    if not names or len(set(names)) != len(names) or set(names) - set(STATISTICAL_HEADS):
        raise ValueError("Choose distinct, supported statistical heads")
    n, d = x.shape
    if n < 2 or ("knn_cosine" in names and n < KNN_NEIGHBORS):
        raise ValueError("Not enough independent training video vectors")
    if "bayesian_gaussian_mixture" in names:
        if validation is None:
            raise ValueError("Mixture selection requires separate model-validation features")
        val = matrix(validation)
        if d != val.shape[1]:
            raise ValueError("Feature dimensions differ")
    mean = x.mean(axis=0)
    delta = x - mean
    scatter = delta.T @ delta
    average_variance = np.trace(scatter) / ((n - 1) * d)
    if average_variance <= 0:
        raise ValueError("Genuine training embeddings have no variance.")

    # IW(nu, Psi) has E[covariance] = Psi / (nu - d - 1).
    nu0 = d + 2
    kappa0 = BAYESIAN_PRIOR_MEAN_WEIGHT
    kappa_n = kappa0 + n
    nu_n = nu0 + n
    psi_n = (nu0 - d - 1) * average_variance * np.eye(d) + scatter
    psi_n += (n * kappa0 / kappa_n) * np.outer(mean, mean)
    heads = {}
    if "mahalanobis" in names:
        ledoit = LedoitWolf().fit(x)
        heads["mahalanobis"] = {"mean": ledoit.location_,
            "cholesky": np.linalg.cholesky(ledoit.covariance_), "shrinkage": ledoit.shrinkage_}
    if "knn_cosine" in names:
        heads["knn_cosine"] = {"training_unit_vectors": unit_rows(x), "neighbors": KNN_NEIGHBORS}
    if "bayesian_gaussian" in names:
        heads["bayesian_gaussian"] = {"mean": n / kappa_n * mean, "kappa_n": kappa_n,
            "nu_n": nu_n, "nu0": nu0, "psi_cholesky": np.linalg.cholesky(psi_n)}
    if "bayesian_gaussian_mixture" in names:
        heads["bayesian_gaussian_mixture"] = fit_mixture(x, val)
    return heads


def unit_rows(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1, norms)


def fit_mixture(x, val):
    # A fixed PCA dimension keeps validation log scores comparable between candidates.
    dimension = min(MIXTURE_PCA_DIM, len(x) - 1, x.shape[1])
    pca = PCA(n_components=dimension, random_state=SEED).fit(x)
    train_z, val_z = pca.transform(x), pca.transform(val)
    components = min(MIXTURE_MAX_COMPONENTS, len(x))
    best = None
    val_nll = {}
    for covariance_type in ("diag", "full"):
        model = BayesianGaussianMixture(n_components=components, covariance_type=covariance_type,
                                        weight_concentration_prior_type="dirichlet_process",
                                        reg_covar=1e-6, max_iter=300, random_state=SEED)
        model.fit(train_z)
        val_nll[covariance_type] = float(-model.score_samples(val_z).mean())
        if best is None or val_nll[covariance_type] < val_nll[best.covariance_type]:
            best = model
    return {
        "pca_mean": pca.mean_, "pca_components": pca.components_, "covariance_type": best.covariance_type,
        "means_": best.means_, "precisions_cholesky_": best.precisions_cholesky_,
        "degrees_of_freedom_": best.degrees_of_freedom_, "mean_precision_": best.mean_precision_,
        "weight_concentration_": np.stack(best.weight_concentration_),
        "n_components": components, "converged": best.converged_,
        "validation_nll_diag": val_nll["diag"], "validation_nll_full": val_nll["full"],
    }


def mixture_scores(x, head):
    model = BayesianGaussianMixture(n_components=int(head["n_components"]),
                                    covariance_type=str(head["covariance_type"]),
                                    weight_concentration_prior_type="dirichlet_process")
    for parameter in ("means_", "precisions_cholesky_", "degrees_of_freedom_", "mean_precision_"):
        setattr(model, parameter, head[parameter])
    model.weight_concentration_ = tuple(head["weight_concentration_"])
    projected = (x - head["pca_mean"]) @ head["pca_components"].T
    return -model.score_samples(projected)


def niw_scores(x, head, posterior_draws=NIW_DRAWS):
    if posterior_draws != 0 and posterior_draws < 2:
        raise ValueError("Use zero draws for predictive scores only, or at least two for uncertainty")
    chol = head["psi_cholesky"]
    centered = solve_triangular(chol, (x - head["mean"]).T, lower=True)
    d, nu = chol.shape[0], int(head["nu_n"])
    kappa = float(head["kappa_n"])
    df = nu - d + 1
    distance = np.sum(centered ** 2, axis=0)
    scale = (kappa + 1) / (kappa * df)
    log_normalizer = (math.lgamma(df / 2) - math.lgamma((df + d) / 2)
                      + 0.5 * d * math.log(df * math.pi * scale) + np.log(chol.diagonal()).sum())
    scores = log_normalizer + 0.5 * (df + d) * np.log1p(distance / (df * scale))
    if posterior_draws == 0:
        return scores, {}

    # Bartlett factor A of the standard Wishart. The sampled precision is
    # L^(-T) A A^T L^(-1); drawing mu conditional on it subtracts iid normals
    # in those whitened coordinates. No inverse or per-draw factorization is needed.
    rng = np.random.default_rng(SEED)
    lower = np.tril_indices(d, -1)
    diag = np.diag_indices(d)
    factor = np.zeros((d, d), order="F")
    mean_distance = np.zeros(len(x))
    sum_squared = np.zeros(len(x))
    for count in range(1, posterior_draws + 1):
        factor[lower] = rng.standard_normal(len(lower[0]))
        factor[diag] = np.sqrt(rng.chisquare(nu - np.arange(d)))
        offset = rng.standard_normal((d, 1)) / math.sqrt(kappa)
        projected = dtrmm(1.0, factor, centered, lower=1, trans_a=1) - offset
        draw = np.einsum("ij,ij->j", projected, projected)
        delta = draw - mean_distance
        mean_distance += delta / count
        sum_squared += delta * (draw - mean_distance)
    return scores, {"posterior_distance_mean": mean_distance,
                    "posterior_distance_std": np.sqrt(np.maximum(sum_squared, 0) / (posterior_draws - 1))}


def score_statistical_head(name, features, head, posterior_draws=NIW_DRAWS):
    x = matrix(features)
    if name == "mahalanobis":
        whitened = solve_triangular(head["cholesky"], (x - head["mean"]).T, lower=True)
        return np.sum(whitened ** 2, axis=0), {}
    if name == "knn_cosine":
        k = int(head["neighbors"])
        distances = []
        for start in range(0, len(x), 256):
            similarities = unit_rows(x[start:start + 256]) @ head["training_unit_vectors"].T
            kth = similarities.shape[1] - k
            distances.append(1 - np.clip(np.partition(similarities, kth, axis=1)[:, kth], -1., 1.))
        return np.concatenate(distances), {}
    if name == "bayesian_gaussian":
        return niw_scores(x, head, posterior_draws)
    if name == "bayesian_gaussian_mixture":
        return mixture_scores(x, head), {}
    raise ValueError(f"Unknown anomaly head {name}")
