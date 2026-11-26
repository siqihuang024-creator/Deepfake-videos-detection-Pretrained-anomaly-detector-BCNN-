"""Small deterministic controls for the existing source-balanced SVDD run."""
import math
import time

import numpy as np
import torch

from . import bayesian_svdd as bayes


def fit_centroid(training, groups):
    weights, count = bayes.source_balanced_weights(groups)
    x = np.asarray(training, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(weights) or not np.isfinite(x).all() or len(x) < 2:
        raise ValueError("Expected finite fitting features and one source ID per row")
    return {"feature_mean": np.sum(x * weights[:, None], axis=0),
            "training_weights": weights, "effective_train_groups": count}


def score_centroid(features, head):
    return np.sum((np.asarray(features, dtype=np.float64) - head["feature_mean"]) ** 2, axis=1)


def point_prior(first, second):
    # The mean-dependent part of the Bayesian Gaussian KL, at the same scale.
    return .5 * (first.square().sum() / (2 / first.shape[1])
                 + second.square().sum() / (2 / second.shape[1]))


def fit_deterministic(training, validation, groups, validation_groups, prior_matched=False, progress=None):
    weights, group_count = bayes.source_balanced_weights(groups)
    validation_weights, validation_count = bayes.source_balanced_weights(validation_groups)
    if set(groups) & set(validation_groups):
        raise ValueError("A source crosses fitting and early stopping")
    x, val = torch.as_tensor(training, dtype=torch.float32), torch.as_tensor(validation, dtype=torch.float32)
    if (x.ndim != 2 or val.ndim != 2 or len(x) < 2 or not len(val) or not x.shape[1]
            or len(x) != len(weights) or len(val) != len(validation_weights)
            or x.shape[1] != val.shape[1] or not torch.isfinite(x).all() or not torch.isfinite(val).all()):
        raise ValueError("Expected finite fitting and independent early-stop features")
    w = torch.as_tensor(weights, dtype=torch.float32)
    vw = torch.as_tensor(validation_weights, dtype=torch.float32)
    w, vw = w / w.sum(), vw / vw.sum()
    started = time.perf_counter()
    feature_mean = (x * w[:, None]).sum(dim=0)
    x, val = x - feature_mean, val - feature_mean
    rng = torch.Generator(device="cpu").manual_seed(bayes.SEED)
    # Exact initial mean matrices and center convention of the Bayesian head.
    first = torch.nn.Parameter(torch.randn((bayes.WIDTHS[0], x.shape[1]), generator=rng) * math.sqrt(2 / x.shape[1]))
    second = torch.nn.Parameter(torch.randn((bayes.WIDTHS[1], bayes.WIDTHS[0]), generator=rng) * math.sqrt(2 / bayes.WIDTHS[0]))
    with torch.no_grad():
        initial = bayes.embed(x, first, second)
        initial_variance, center = bayes._weighted_moments(initial, w)
        initial_variance = float(initial_variance.mean())
        center = torch.where(center.abs() < bayes.CENTER_EPS,
                             torch.where(center < 0, -bayes.CENTER_EPS, bayes.CENTER_EPS), center)
    optimizer = torch.optim.Adam([first, second], lr=bayes.LEARNING_RATE)
    best, history, best_value = {}, [], math.inf
    for epoch in range(1, bayes.MAX_EPOCHS + 1):
        totals = np.zeros(3, dtype=float)
        for indices in torch.randperm(len(x), generator=rng).split(bayes.BATCH_SIZE):
            distance = bayes.squared_distances(x[indices], first, second, center)
            data_loss = (distance * w[indices] * len(x)).mean()
            regularizer = bayes.KL_WEIGHT * point_prior(first, second) / group_count if prior_matched else data_loss.new_zeros(())
            loss = data_loss + regularizer
            if not torch.isfinite(loss):
                raise ValueError("Deterministic SVDD training diverged")
            totals += np.asarray([float(loss.detach()), float(data_loss.detach()), float(regularizer.detach())]) * len(indices)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            objective = float((bayes.squared_distances(val, first, second, center) * vw).sum())
        if not math.isfinite(objective):
            raise ValueError("Nonfinite early-stop objective")
        if objective < best_value:
            best_value = objective
            best = {"best_epoch": epoch, "validation_distance": objective,
                    "first_mean": first.detach().numpy().copy(), "second_mean": second.detach().numpy().copy()}
        history.append([epoch, *(totals / len(x)), objective, time.perf_counter() - started])
        if progress:
            progress({"epoch": epoch, "training_loss": totals[0] / len(x), "data_loss": totals[1] / len(x),
                      "regularizer": totals[2] / len(x), "early_stop_distance": objective, "best_epoch": best["best_epoch"]})
        if epoch - best["best_epoch"] >= bayes.PATIENCE:
            break
    head = {**best, "feature_mean": feature_mean.numpy(), "center": center.numpy(), "training_weights": w.numpy(),
            "validation_weights": vw.numpy(), "effective_train_groups": group_count,
            "early_stop_source_groups": validation_count, "prior_matched": prior_matched,
            "training_history": np.asarray(history), "epochs_run": epoch, "training_seconds": time.perf_counter() - started}
    with torch.no_grad():
        scores = score_deterministic(training, head)
        mean = float(np.sum(scores * weights))
        relative_std = float(np.sqrt(np.sum((scores - mean) ** 2 * weights)) / mean) if mean > 0 else 0.
        embeddings = bayes.embed(x, torch.as_tensor(head["first_mean"]), torch.as_tensor(head["second_mean"]))
        variance, _ = bayes._weighted_moments(embeddings, w)
        ratio = float(variance.mean()) / initial_variance if initial_variance > 0 else 0.
    head.update(train_score_relative_std=relative_std, train_embedding_variance_ratio=ratio,
                near_constant_train_scores=relative_std <= bayes.MIN_TRAIN_SCORE_RELATIVE_STD,
                low_train_embedding_variance=ratio <= bayes.MIN_TRAIN_EMBEDDING_VARIANCE_RATIO)
    return head


@torch.no_grad()
def score_deterministic(features, head):
    x = torch.as_tensor(features, dtype=torch.float32) - torch.as_tensor(head["feature_mean"])
    result = bayes.squared_distances(x, torch.as_tensor(head["first_mean"]),
                                   torch.as_tensor(head["second_mean"]), torch.as_tensor(head["center"]))
    return result.numpy()
