"""Bias-free Bayesian Deep SVDD trained on genuine video features."""

import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .anomaly import SEED

WIDTHS = (512, 128)
CENTER_EPS = 0.1
KL_WEIGHT = 1e-3
MAX_EPOCHS = 100
PATIENCE = 12
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
VALIDATION_DRAWS = 8
POSTERIOR_DRAWS = 50
MIN_TRAIN_SCORE_RELATIVE_STD = 1e-3
MIN_TRAIN_EMBEDDING_VARIANCE_RATIO = 1e-4


def source_balanced_weights(groups):
    """Equal total weight per supplied source family; clips remain correlated."""
    groups = np.asarray(groups, dtype=object)
    if (groups.ndim != 1 or not len(groups)
            or any(not isinstance(value, (str, np.str_)) or not value.strip() for value in groups)):
        raise ValueError("Nonempty source-family IDs required for every feature vector")
    names, inverse, counts = np.unique(groups.astype(str), return_inverse=True, return_counts=True)
    return 1. / (len(names) * counts[inverse]), len(names)


def _normalized_weights(values, count, device):
    values = torch.as_tensor(values, dtype=torch.float32, device=device).detach()
    if (values.shape != (count,) or not torch.isfinite(values).all() or not (values > 0).all()
            or not torch.isfinite(values.sum())):
        raise ValueError("Expected a finite positive weight for every feature vector")
    return values / values.sum()


def _weighted_moments(values, weights):
    mean = (values * weights[:, None]).sum(dim=0)
    variance = ((values - mean).square() * weights[:, None]).sum(dim=0)
    return variance, mean


class BayesianLinear(nn.Module):
    def __init__(self, inputs, outputs, generator, device):
        super().__init__()
        self.prior_std = math.sqrt(2 / inputs)
        initial_std = self.prior_std * 0.15
        shape = (outputs, inputs)
        self.mean = nn.Parameter(torch.randn(shape, generator=generator, device=device) * self.prior_std)
        self.rho = nn.Parameter(torch.full(shape, math.log(math.expm1(initial_std)), device=device))

    def sample(self, generator):
        noise = torch.randn(self.mean.shape, generator=generator, device=self.mean.device)
        return self.mean + F.softplus(self.rho) * noise

    def kl(self):
        std = F.softplus(self.rho)
        scaled = (self.mean.square() + std.square()) / self.prior_std ** 2
        return (0.5 * (scaled - 1) + math.log(self.prior_std) - std.log()).sum()


def embed(x, first, second):
    activations = F.leaky_relu(F.linear(x, first), negative_slope=0.01)
    return F.linear(activations, second)


def squared_distances(x, first, second, center):
    return (embed(x, first, second) - center).square().sum(dim=1)


@torch.no_grad()
def posterior_distances(x, center, sample_weights, draws, aggregation=None):
    """Use each sampled network for every video, independent of batching."""
    if draws < 2:
        raise ValueError("At least two posterior draws required for sample variance")
    if aggregation is not None:
        inverse, weights, group_count = aggregation
        if (inverse.shape != (len(x),) or weights.shape != (len(x),) or group_count < 1
                or inverse.dtype != torch.long or inverse.device != x.device or weights.device != x.device
                or (inverse < 0).any() or (inverse >= group_count).any()
                or not torch.isfinite(weights).all() or not (weights > 0).all()):
            raise ValueError("Invalid video aggregation indices or weights")
        totals = torch.zeros(group_count, device=x.device).index_add_(0, inverse, weights)
        if not torch.allclose(totals, torch.ones_like(totals), rtol=1e-5, atol=1e-6):
            raise ValueError("Clip weights must sum to one inside every video")
    mean = torch.zeros(len(x) if aggregation is None else group_count, device=x.device)
    sum_squared = torch.zeros_like(mean)
    for count in range(1, draws + 1):
        first, second = sample_weights()
        distance = torch.cat([squared_distances(batch, first, second, center) for batch in x.split(BATCH_SIZE)])
        if aggregation is not None:
            # Aggregate each SHARED network draw before estimating variance,
            # preserving covariance between windows from the same video.
            distance = torch.zeros(group_count, device=x.device).index_add_(0, inverse, distance * weights)
        delta = distance - mean
        mean += delta / count
        sum_squared += delta * (distance - mean)
    return mean, sum_squared.clamp_min(0) / (draws - 1)


def fit(training, validation, device, progress=None, *, training_weights=None,
        validation_weights=None, effective_train_groups=None):
    device = torch.device(device)
    x = torch.as_tensor(training, dtype=torch.float32, device=device)
    val = torch.as_tensor(validation, dtype=torch.float32, device=device)
    if (x.ndim != 2 or val.ndim != 2 or len(x) < 2 or not len(val)
            or x.shape[1] != val.shape[1] or not x.shape[1]
            or not torch.isfinite(x).all() or not torch.isfinite(val).all()):
        raise ValueError("Expected finite genuine fitting and independent early-stop feature matrices")
    balanced = training_weights is not None or validation_weights is not None or effective_train_groups is not None
    train_weights, val_weights = None, None
    denominator = len(x)
    if balanced:
        if training_weights is None or validation_weights is None or effective_train_groups is None:
            raise ValueError("Weighted fitting requires both weight vectors and an explicit source-group count")
        if (isinstance(effective_train_groups, (bool, np.bool_))
                or not isinstance(effective_train_groups, (int, np.integer))
                or not 1 <= effective_train_groups <= len(x)):
            raise ValueError("Source-group count must be an integer between one and the number of vectors")
        train_weights = _normalized_weights(training_weights, len(x), device)
        val_weights = _normalized_weights(validation_weights, len(val), device)
        denominator = int(effective_train_groups)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    feature_mean = x.mean(dim=0) if train_weights is None else (x * train_weights[:, None]).sum(dim=0)
    x = x - feature_mean
    val = val - feature_mean
    train_rng = torch.Generator(device=device).manual_seed(SEED)
    first = BayesianLinear(x.shape[1], WIDTHS[0], train_rng, device)
    second = BayesianLinear(WIDTHS[0], WIDTHS[1], train_rng, device)
    with torch.no_grad():
        initial_embeddings = torch.cat([embed(batch, first.mean, second.mean) for batch in x.split(BATCH_SIZE)])
        if train_weights is None:
            initial_variance, center = torch.var_mean(initial_embeddings, dim=0, unbiased=False)
        else:
            initial_variance, center = _weighted_moments(initial_embeddings, train_weights)
        initial_variance = float(initial_variance.mean())
        center = torch.where(center.abs() < CENTER_EPS, torch.where(center < 0, -CENTER_EPS, CENTER_EPS), center)
        del initial_embeddings

    optimizer = torch.optim.Adam([*first.parameters(), *second.parameters()], lr=LEARNING_RATE)
    best_distance = math.inf
    best = {}
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        training_loss = 0.
        for indices in torch.randperm(len(x), generator=train_rng, device=device).split(BATCH_SIZE):
            distance = squared_distances(x[indices], first.sample(train_rng), second.sample(train_rng), center)
            data_loss = (distance.mean() if train_weights is None else
                         (distance * train_weights[indices] * len(x)).mean())
            loss = data_loss + KL_WEIGHT * (first.kl() + second.kl()) / denominator
            if not torch.isfinite(loss):
                raise ValueError(f"Bayesian Deep SVDD training diverged at epoch {epoch}")
            training_loss += float(loss.detach()) * len(indices)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # Common random numbers reduce noise when comparing checkpoints on A.
        val_rng = torch.Generator(device=device).manual_seed(SEED + 1)
        distances, _ = posterior_distances(
            val, center, lambda: (first.sample(val_rng), second.sample(val_rng)), VALIDATION_DRAWS
        )
        objective = float(distances.mean() if val_weights is None else (distances * val_weights).sum())
        if not math.isfinite(objective):
            raise ValueError(f"Bayesian Deep SVDD validation diverged at epoch {epoch}.")
        if objective < best_distance:
            best_distance = objective
            best = {
                "best_epoch": epoch, "validation_distance": objective,
                "first_mean": first.mean.detach().cpu().numpy().copy(),
                "second_mean": second.mean.detach().cpu().numpy().copy(),
                "first_std": F.softplus(first.rho).detach().cpu().numpy(),
                "second_std": F.softplus(second.rho).detach().cpu().numpy(),
            }
        elapsed = time.perf_counter() - started
        history.append([epoch, training_loss / len(x), objective, elapsed])
        if progress is not None:
            progress({"epoch": epoch, "training_loss": training_loss / len(x),
                      "early_stop_distance": objective, "best_epoch": best["best_epoch"], "seconds": elapsed})
        if epoch - best["best_epoch"] >= PATIENCE:
            break
    head = {**best, "epochs_run": epoch, "center": center.cpu().numpy(), "feature_mean": feature_mean.cpu().numpy(),
            "training_history": np.asarray(history, dtype=np.float64),
            "training_seconds": time.perf_counter() - started}
    if balanced:
        head.update(training_weights=train_weights.cpu().numpy(), validation_weights=val_weights.cpu().numpy(),
                    effective_train_groups=denominator)
    return {**head, **training_diagnostics(x, head, initial_variance, train_weights)}


def fit_source_balanced(training, validation, training_groups, validation_groups, device, progress=None):
    """Source-balanced surrogate objective with fixed source-count KL scaling.

    This does not turn correlated windows into independent observations or
    guarantee calibrated posterior uncertainty. Formal callers must register
    their group IDs, splits and this weighting/KL recipe before evaluation.
    """
    train_weights, groups = source_balanced_weights(training_groups)
    val_weights, validation_count = source_balanced_weights(validation_groups)
    if set(training_groups) & set(validation_groups):
        raise ValueError("A source family crosses gradient fitting and early stopping")
    head = fit(training, validation, device, progress, training_weights=train_weights,
               validation_weights=val_weights, effective_train_groups=groups)
    head["early_stop_source_groups"] = validation_count
    return head


def score_centered(x, head, aggregation=None):
    device = x.device
    center = torch.as_tensor(head["center"], device=device)
    means = [torch.as_tensor(head[layer + "_mean"], device=device) for layer in ("first", "second")]
    scales = [torch.as_tensor(head[layer + "_std"], device=device) for layer in ("first", "second")]
    rng = torch.Generator(device=device).manual_seed(SEED + 2)

    def sample_weights():
        return [mean + std * torch.randn(mean.shape, generator=rng, device=device)
                for mean, std in zip(means, scales)]

    return posterior_distances(x, center, sample_weights, POSTERIOR_DRAWS, aggregation)


@torch.no_grad()
def training_diagnostics(x, head, initial_variance, weights=None):
    distance, _ = score_centered(x, head)
    if weights is None:
        variance, mean = torch.var_mean(distance, unbiased=False)
    else:
        mean = (distance * weights).sum()
        variance = ((distance - mean).square() * weights).sum()
    variance, mean = float(variance), float(mean)
    if not math.isfinite(variance) or not math.isfinite(mean):
        raise ValueError("Bayesian Deep SVDD produced non-finite training scores.")
    relative_std = math.sqrt(variance) / mean if mean > 0 else 0.0
    first = torch.as_tensor(head["first_mean"], device=x.device)
    second = torch.as_tensor(head["second_mean"], device=x.device)
    embeddings = torch.cat([embed(batch, first, second) for batch in x.split(BATCH_SIZE)])
    embedding_variance = float(embeddings.var(dim=0, unbiased=False).mean() if weights is None else
                               _weighted_moments(embeddings, weights)[0].mean())
    embedding_variance_ratio = embedding_variance / initial_variance if initial_variance > 0 else 0.0
    return {
        "train_score_mean": mean,
        "train_score_variance": variance,
        "train_score_relative_std": relative_std,
        "near_constant_train_scores": relative_std <= MIN_TRAIN_SCORE_RELATIVE_STD,
        "initial_train_embedding_variance": initial_variance,
        "train_embedding_variance": embedding_variance,
        "train_embedding_variance_ratio": embedding_variance_ratio,
        "low_train_embedding_variance": embedding_variance_ratio <= MIN_TRAIN_EMBEDDING_VARIANCE_RATIO,
    }


def score(features, head, device):
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    if "feature_mean" in head:
        x = x - torch.as_tensor(head["feature_mean"], device=device)
    distance, variance = score_centered(x, head)
    return distance.cpu().numpy(), {"posterior_distance_variance": variance.cpu().numpy()}


def score_grouped(features, head, groups, device, clip_weights=None):
    """Return a mean-distance score and posterior variance per supplied video.

    Equal clip weights by default; callers may supply positive duration weights.
    Aggregation is inside every shared posterior draw, not a mean of variances.
    This API does not select clips or choose a calibration protocol.
    """
    source_balanced_weights(groups)  # Validate supplied IDs, including missing/empty values.
    names, inverse = np.unique(np.asarray(groups, dtype=str), return_inverse=True)
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    if x.ndim != 2 or len(x) != len(inverse) or not torch.isfinite(x).all():
        raise ValueError("Finite features and one video ID per feature vector required")
    weights = np.ones(len(x), dtype=np.float64) if clip_weights is None else np.asarray(clip_weights, dtype=np.float64)
    if weights.shape != (len(x),) or not np.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError("Finite positive aggregation weights required")
    totals = np.bincount(inverse, weights=weights)
    if not np.isfinite(totals).all():
        raise ValueError("Non-finite total aggregation weight")
    weights = weights / totals[inverse]
    aggregation = (torch.as_tensor(inverse, dtype=torch.long, device=x.device),
                   torch.as_tensor(weights, dtype=torch.float32, device=x.device), len(names))
    if "feature_mean" in head:
        x = x - torch.as_tensor(head["feature_mean"], device=x.device)
    distance, variance = score_centered(x, head, aggregation)
    return names, distance.cpu().numpy(), {"posterior_distance_variance": variance.cpu().numpy()}
