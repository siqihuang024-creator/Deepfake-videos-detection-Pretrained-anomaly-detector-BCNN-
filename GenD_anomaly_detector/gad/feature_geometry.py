"""Source-balanced shrinkage distances and classifier-direction diagnostics."""
import numpy as np
from scipy.linalg import solve_triangular
from sklearn.covariance import LedoitWolf


def matrix(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError("Expected nonempty finite feature matrix")
    return x


def source_means(features, groups):
    x = matrix(features)
    groups = np.asarray(groups, dtype=object)
    if (groups.shape != (len(x),) or any(not isinstance(g, str) or not g.strip() for g in groups)):
        raise ValueError("One nonempty source ID required per feature row")
    names, inverse, counts = np.unique(groups.astype(str), return_inverse=True, return_counts=True)
    sums = np.zeros((len(names), x.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, x)
    return names, sums / counts[:, None]


def fit_distances(features, groups):
    names, means = source_means(features, groups)
    if len(means) < 2 or np.var(means, axis=0).mean() <= 0:
        raise ValueError("At least two varying source means required for covariance estimation")
    model = LedoitWolf(store_precision=False).fit(means)
    covariance = model.covariance_
    if not np.isfinite(covariance).all():
        raise ValueError("Nonfinite source-mean covariance")
    try:
        cholesky = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Shrinkage covariance is not positive definite; inspect fitting-source variation") from exc
    return {"mean": model.location_, "cholesky": cholesky,
            "shrinkage": float(model.shrinkage_), "source_count": len(names),
            "fit_clip_count": len(features), "covariance_trace": float(np.trace(covariance))}


def score_distances(features, head):
    delta = matrix(features) - head["mean"]
    z = solve_triangular(head["cholesky"], delta.T, lower=True)
    return np.sum(delta * delta, axis=1), np.sum(z * z, axis=0)


def classifier_direction(weights):
    w = matrix(weights)
    if w.shape[0] != 2:
        raise ValueError("Expected two-class linear weights, real=0 and fake=1")
    vector = w[1] - w[0]
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Classifier direction is degenerate")
    return vector / norm


def remove_direction(features, direction):
    """Householder coordinates for the orthogonal complement; no renormalization.

    The map is determined by frozen classifier weights only. Its output has
    d-1 orthonormal coordinates, avoiding a redundant zero covariance axis.
    """
    x = matrix(features)
    u = np.asarray(direction, dtype=float)
    if u.shape != (x.shape[1],) or len(u) < 2 or not np.isfinite(u).all() or not np.isclose(np.linalg.norm(u), 1):
        raise ValueError("A finite unit classifier direction with matching dimension is required")
    v = u.copy()
    v[0] += 1 if u[0] >= 0 else -1
    v /= np.linalg.norm(v)
    return (x - 2 * (x @ v)[:, None] * v)[..., 1:]


def squared_pairs(x, y):
    x, y = matrix(x), matrix(y)
    return np.maximum((x*x).sum(1)[:, None] + (y*y).sum(1)[None, :] - 2*x@y.T, 0.)


def real_feature_shift(features, rows):
    """Descriptive source-mean comparisons; not a trained domain classifier."""
    x = matrix(features)
    if len(rows) != len(x):
        raise ValueError("Feature and metadata lengths differ")
    sets = {}
    for name, mask in (
        ("A_fit", rows.head_role.eq("gradient_fit")),
        ("A_calibration", rows.head_role.eq("calibration")),
        ("DFDC_real", rows.head_role.eq("dev") & rows.label.eq(0)),
    ):
        _, sets[name] = source_means(x[mask], rows.loc[mask, "source_clip"].tolist())
    fit = sets["A_fit"]
    distances = squared_pairs(fit, fit)
    off_diagonal = distances[np.triu_indices(len(fit), 1)]
    positive = off_diagonal[off_diagonal > 0]
    if not len(positive):
        raise ValueError("No positive fitting-source separation to set the kernel scale")
    gamma = 1 / (2 * float(np.median(positive)))
    results = []
    for a, b in (("A_fit", "A_calibration"), ("A_fit", "DFDC_real"), ("A_calibration", "DFDC_real")):
        left, right = sets[a], sets[b]
        mmd = (np.exp(-gamma * squared_pairs(left, left)).mean()
               + np.exp(-gamma * squared_pairs(right, right)).mean()
               - 2*np.exp(-gamma * squared_pairs(left, right)).mean())
        results.append({"left": a, "right": b, "left_sources": len(left), "right_sources": len(right),
                        "mean_vector_distance": float(np.linalg.norm(left.mean(0)-right.mean(0))),
                        "rbf_mmd_squared_biased": float(max(0., mmd))})
    return {"kernel_gamma_from_fitting_sources_only": gamma, "comparisons": results,
            "limitation": "Descriptive biased MMD; unequal finite sample sizes affect it. No p-value, person-independence or causal domain-shift claim."}
