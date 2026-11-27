"""Fixed source-balanced clip covariance and explicit distance decompositions."""
import numpy as np
from scipy.linalg import solve_triangular

from .feature_geometry import matrix, source_means


SHRINKAGES = (.3, .5)


def clip_covariance(features, groups):
    """Biased population covariance: every source has total weight 1/G.

    This estimates a source-balanced mixture of clips, not G independent
    observations of individual clips. No iid sample-size correction is used.
    """
    x = matrix(features)
    names, means = source_means(x, groups)
    if len(names) < 2:
        raise ValueError("At least two fitting sources required")
    _, inverse, counts = np.unique(np.asarray(groups, str), return_inverse=True, return_counts=True)
    weights = 1. / (len(names) * counts[inverse])
    mean = means.mean(0)
    delta = x - mean
    covariance = (delta * weights[:, None]).T @ delta
    centered = means - mean
    between = centered.T @ centered / len(names)
    within_delta = x - means[inverse]
    within = (within_delta * weights[:, None]).T @ within_delta
    np.testing.assert_allclose(covariance, between + within, rtol=1e-8, atol=1e-12)
    return mean, covariance, {"sources": len(names), "clips": len(x),
        "between_trace": float(np.trace(between)), "within_trace": float(np.trace(within)),
        "total_trace": float(np.trace(covariance))}


def spectrum(empirical):
    c = matrix(empirical)
    if c.shape[0] != c.shape[1] or not np.allclose(c, c.T, rtol=1e-9, atol=1e-12):
        raise ValueError("Expected symmetric covariance")
    eigenvalues, vectors = np.linalg.eigh(c)
    scale = max(float(eigenvalues[-1]), np.finfo(float).tiny)
    tolerance = max(1e-12 * scale, 10 * np.finfo(float).eps * len(c) * scale)
    if eigenvalues[0] < -tolerance or np.trace(c) <= 0:
        raise ValueError("Invalid empirical covariance spectrum")
    null = eigenvalues <= tolerance
    return eigenvalues, vectors[:, null], tolerance


def regularize(mean, empirical, shrinkage, eigensystem=None):
    if not np.isfinite(shrinkage) or not 0 < shrinkage <= 1:
        raise ValueError("Shrinkage must lie in (0, 1]")
    eigenvalues, null, tolerance = spectrum(empirical) if eigensystem is None else eigensystem
    d = len(mean)
    floor = float(shrinkage * np.trace(empirical) / d)
    covariance = (1-shrinkage) * empirical + floor * np.eye(d)
    chol = np.linalg.cholesky(covariance)
    return {"mean": np.asarray(mean, float), "cholesky": chol, "shrinkage": float(shrinkage),
        "floor": floor, "empirical_eigenvalues": eigenvalues, "empirical_null_basis": null,
        "null_tolerance": float(tolerance),
        "condition_number": float(((1-shrinkage)*eigenvalues[-1]+floor)
                                  / ((1-shrinkage)*max(0., eigenvalues[0])+floor))}


def recover_source_mean_head(saved):
    """Recover the unshrunk geometry-342 covariance without refitting it."""
    chol = saved["cholesky"]
    covariance = chol @ chol.T
    shrinkage = float(saved["shrinkage"])
    if not 0 < shrinkage < 1:
        raise ValueError("Cannot recover empirical covariance for this shrinkage")
    floor = shrinkage * np.trace(covariance) / len(covariance)
    empirical = (covariance - floor*np.eye(len(covariance))) / (1-shrinkage)
    head = regularize(saved["mean"], empirical, shrinkage)
    np.testing.assert_allclose(head["cholesky"], chol, rtol=1e-7, atol=1e-10)
    return head


def distance_parts(features, head, reference_null):
    """Own-null energy plus a fixed OLD-null split, including signed cross term.

    New covariance can mix the old null/support spaces. Therefore the old-null
    self quadratic divided by total is NOT a bounded fraction. Only the current
    empirical-null spectral contribution is a nonnegative additive share.
    """
    delta = matrix(features) - head["mean"]
    total = (solve_triangular(head["cholesky"], delta.T, lower=True)**2).sum(0)
    null_energy = ((delta @ head["empirical_null_basis"])**2).sum(1) / head["floor"]
    old_null_delta = (delta @ reference_null) @ reference_null.T
    old_null_self = (solve_triangular(head["cholesky"], old_null_delta.T, lower=True)**2).sum(0)
    support_self = (solve_triangular(head["cholesky"], (delta-old_null_delta).T, lower=True)**2).sum(0)
    cross = total - old_null_self - support_self
    if np.any(null_energy > total + 1e-7*np.maximum(1., total)):
        raise ValueError("Nullspace energy exceeds total quadratic distance")
    share = np.divide(null_energy, total, out=np.zeros_like(total), where=total > 0)
    return {"score": total, "own_null_contribution": null_energy, "own_null_share": share,
        "old_null_self": old_null_self, "old_support_self": support_self, "old_cross": cross}
