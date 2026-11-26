"""Instrumented KL-weight ablation; preserves the archived Bayesian algorithm file."""
import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from . import bayesian_svdd as original

VARIANTS = {"kl_1e-4": 1e-4, "kl_1e-5": 1e-5}
HISTORY_COLUMNS = ("epoch", "training_loss", "training_distance_loss", "training_kl_raw",
                   "training_weighted_kl", "early_stop_distance", "elapsed_seconds")


def fit(training, validation, groups, validation_groups, kl_weight, progress=None):
    """Same sampled CPU optimizer trajectory as original.fit at the same beta.

    Extra logs average loss components from the SAME pre-update minibatch states.
    No early-stop distance is substituted for the training distance term.
    """
    if isinstance(kl_weight, (bool, np.bool_)) or not isinstance(kl_weight, (float, int)) or not math.isfinite(kl_weight) or kl_weight <= 0:
        raise ValueError("KL weight must be finite and positive")
    weights, count = original.source_balanced_weights(groups)
    validation_weights, validation_count = original.source_balanced_weights(validation_groups)
    if set(groups) & set(validation_groups):
        raise ValueError("A source crosses fitting and early stopping")
    device = torch.device("cpu")
    x, val = torch.as_tensor(training, dtype=torch.float32), torch.as_tensor(validation, dtype=torch.float32)
    if (x.ndim != 2 or val.ndim != 2 or len(x) < 2 or not len(val) or not x.shape[1]
            or x.shape[1] != val.shape[1] or len(x) != len(weights) or len(val) != len(validation_weights)
            or not torch.isfinite(x).all() or not torch.isfinite(val).all()):
        raise ValueError("Expected finite fitting and separate early-stop features")
    w = original._normalized_weights(weights, len(x), device)
    vw = original._normalized_weights(validation_weights, len(val), device)
    started = time.perf_counter()
    feature_mean = (x * w[:, None]).sum(dim=0)
    x, val = x - feature_mean, val - feature_mean
    rng = torch.Generator(device=device).manual_seed(original.SEED)
    first = original.BayesianLinear(x.shape[1], original.WIDTHS[0], rng, device)
    second = original.BayesianLinear(original.WIDTHS[0], original.WIDTHS[1], rng, device)
    with torch.no_grad():
        initial = torch.cat([original.embed(batch, first.mean, second.mean) for batch in x.split(original.BATCH_SIZE)])
        variance, center = original._weighted_moments(initial, w)
        initial_variance = float(variance.mean())
        center = torch.where(center.abs() < original.CENTER_EPS,
                             torch.where(center < 0, -original.CENTER_EPS, original.CENTER_EPS), center)
        del initial
    optimizer = torch.optim.Adam([*first.parameters(), *second.parameters()], lr=original.LEARNING_RATE)
    best, history, best_value = {}, [], math.inf
    for epoch in range(1, original.MAX_EPOCHS + 1):
        totals = np.zeros(4, dtype=float)
        for indices in torch.randperm(len(x), generator=rng, device=device).split(original.BATCH_SIZE):
            distance = original.squared_distances(x[indices], first.sample(rng), second.sample(rng), center)
            data_loss = (distance * w[indices] * len(x)).mean()
            raw_kl = first.kl() + second.kl()
            weighted_kl = kl_weight * raw_kl / count
            loss = data_loss + weighted_kl
            if not torch.isfinite(loss):
                raise ValueError("Bayesian KL ablation diverged")
            totals += np.asarray([float(v.detach()) for v in (loss, data_loss, raw_kl, weighted_kl)]) * len(indices)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            validation_rng = torch.Generator(device=device).manual_seed(original.SEED + 1)
            distances, _ = original.posterior_distances(val, center,
                lambda: (first.sample(validation_rng), second.sample(validation_rng)), original.VALIDATION_DRAWS)
            objective = float((distances * vw).sum())
        if not math.isfinite(objective):
            raise ValueError("Nonfinite Bayesian early-stop objective")
        if objective < best_value:
            best_value = objective
            best = {"best_epoch": epoch, "validation_distance": objective,
                    "first_mean": first.mean.detach().numpy().copy(), "second_mean": second.mean.detach().numpy().copy(),
                    "first_std": F.softplus(first.rho).detach().numpy().copy(),
                    "second_std": F.softplus(second.rho).detach().numpy().copy()}
        means = totals / len(x)
        elapsed = time.perf_counter() - started
        history.append([epoch, *means, objective, elapsed])
        if progress:
            progress({"epoch": epoch, **dict(zip(HISTORY_COLUMNS[1:5], means)),
                      "training_kl_per_source": means[2] / count, "early_stop_distance": objective,
                      "best_epoch": best["best_epoch"], "seconds": elapsed})
        if epoch - best["best_epoch"] >= original.PATIENCE:
            break
    head = {**best, "epochs_run": epoch, "center": center.numpy(), "feature_mean": feature_mean.numpy(),
            "training_history": np.asarray(history), "training_seconds": time.perf_counter() - started,
            "training_weights": w.numpy(), "validation_weights": vw.numpy(), "effective_train_groups": count,
            "early_stop_source_groups": validation_count, "kl_weight": kl_weight}
    head.update(original.training_diagnostics(x, head, initial_variance, w))
    raw_kl = 0.
    for layer in ("first", "second"):
        mean, std = head[layer + "_mean"].astype(float), head[layer + "_std"].astype(float)
        prior_variance = 2 / mean.shape[1]
        raw_kl += float(.5 * np.sum((mean * mean + std * std) / prior_variance - 1
                                    + np.log(prior_variance) - 2 * np.log(std)))
        head[layer + "_mean_abs"] = float(np.abs(mean).mean())
        head[layer + "_std_mean"] = float(std.mean())
        head[layer + "_median_abs_mean_over_std"] = float(np.median(np.abs(mean) / std))
    head["best_state_kl_raw"] = raw_kl
    head["best_state_weighted_kl"] = kl_weight * raw_kl / count
    return head
