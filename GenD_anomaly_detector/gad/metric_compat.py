"""Finite binary EER thresholds without changing the pinned upstream checkout."""

import numpy as np


def calculate_eer(labels, probabilities, return_threshold=False):
    """Interpolate the empirical ROC, never interpolate its infinite endpoint.

    Tied scores need not admit a deterministic threshold attaining the
    interpolated EER. The returned finite threshold is descriptive, not a
    threshold calibrated on a separate dataset.
    """
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    if (labels.ndim != 1 or probabilities.shape != (len(labels), 2)
            or set(np.unique(labels)) != {0, 1} or not np.isfinite(probabilities).all()):
        raise ValueError("EER requires finite two-class probabilities and both labels")
    scores = probabilities[:, 1].astype(np.float64)
    order = np.argsort(-scores, kind="stable")
    ordered, positive = scores[order], labels[order] == 1
    ends = np.r_[np.flatnonzero(np.diff(ordered)), len(ordered) - 1]
    tp = np.r_[0., np.cumsum(positive)[ends]].astype(float)
    fp = np.r_[0., 1 + ends - np.cumsum(positive)[ends]].astype(float)
    tpr, fpr = tp / positive.sum(), fp / (~positive).sum()
    # Upstream classifies with score > threshold. nextafter includes a tied
    # score group at each finite ROC endpoint under that strict comparison.
    thresholds = np.r_[ordered[0], np.nextafter(ordered[ends], -np.inf)]
    difference = fpr + tpr - 1
    right = int(np.flatnonzero(difference >= 0)[0])
    left = max(0, right - 1)
    fraction = (0. if difference[right] == difference[left]
                else -difference[left] / (difference[right] - difference[left]))
    eer = float(fpr[left] + fraction * (fpr[right] - fpr[left]))
    if eer == 0:
        threshold = float((scores[labels == 0].max() + scores[labels == 1].min()) / 2)
    else:
        threshold = float(thresholds[left] + fraction * (thresholds[right] - thresholds[left]))
    if not np.isfinite(threshold):
        raise ValueError("Failed to produce a finite EER threshold")
    return (eer, threshold) if return_threshold else eer


def install_upstream_fix():
    """Patch the module object used by upstream logging, not its source files."""
    from src import metrics
    metrics.calculate_eer = calculate_eer
