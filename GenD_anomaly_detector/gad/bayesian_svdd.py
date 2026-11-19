"""Bias-free Bayesian Deep SVDD trained on genuine video features."""

import math

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
def posterior_distances(x, center, sample_weights, draws):
    """Use each sampled network for every video, independent of batching."""
    mean = torch.zeros(len(x), device=x.device)
    sum_squared = torch.zeros_like(mean)
    for count in range(1, draws + 1):
        first, second = sample_weights()
        distance = torch.cat([squared_distances(batch, first, second, center) for batch in x.split(BATCH_SIZE)])
        delta = distance - mean
        mean += delta / count
        sum_squared += delta * (distance - mean)
    return mean, sum_squared.clamp_min(0) / (draws - 1)


def fit(training, validation, device):
    x = torch.as_tensor(training, dtype=torch.float32, device=device)
    feature_mean = x.mean(dim=0)
    x = x - feature_mean
    val = torch.as_tensor(validation, dtype=torch.float32, device=device) - feature_mean
    train_rng = torch.Generator(device=device).manual_seed(SEED)
    first = BayesianLinear(x.shape[1], WIDTHS[0], train_rng, device)
    second = BayesianLinear(WIDTHS[0], WIDTHS[1], train_rng, device)
    with torch.no_grad():
        initial_embeddings = torch.cat([embed(batch, first.mean, second.mean) for batch in x.split(BATCH_SIZE)])
        initial_variance, center = torch.var_mean(initial_embeddings, dim=0, unbiased=False)
        initial_variance = float(initial_variance.mean())
        center = torch.where(center.abs() < CENTER_EPS, torch.where(center < 0, -CENTER_EPS, CENTER_EPS), center)
        del initial_embeddings

    optimizer = torch.optim.Adam([*first.parameters(), *second.parameters()], lr=LEARNING_RATE)
    best_distance = math.inf
    best = {}
    for epoch in range(1, MAX_EPOCHS + 1):
        for indices in torch.randperm(len(x), generator=train_rng, device=device).split(BATCH_SIZE):
            distance = squared_distances(x[indices], first.sample(train_rng), second.sample(train_rng), center)
            loss = distance.mean() + KL_WEIGHT * (first.kl() + second.kl()) / len(x)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # Common random numbers reduce noise when comparing checkpoints on A.
        val_rng = torch.Generator(device=device).manual_seed(SEED + 1)
        distances, _ = posterior_distances(
            val, center, lambda: (first.sample(val_rng), second.sample(val_rng)), VALIDATION_DRAWS
        )
        objective = float(distances.mean())
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
        if epoch - best["best_epoch"] >= PATIENCE:
            break
    head = {**best, "epochs_run": epoch, "center": center.cpu().numpy(), "feature_mean": feature_mean.cpu().numpy()}
    return {**head, **training_diagnostics(x, head, initial_variance)}


def score_centered(x, head):
    device = x.device
    center = torch.as_tensor(head["center"], device=device)
    means = [torch.as_tensor(head[layer + "_mean"], device=device) for layer in ("first", "second")]
    scales = [torch.as_tensor(head[layer + "_std"], device=device) for layer in ("first", "second")]
    rng = torch.Generator(device=device).manual_seed(SEED + 2)

    def sample_weights():
        return [mean + std * torch.randn(mean.shape, generator=rng, device=device)
                for mean, std in zip(means, scales)]

    return posterior_distances(x, center, sample_weights, POSTERIOR_DRAWS)


@torch.no_grad()
def training_diagnostics(x, head, initial_variance):
    distance, _ = score_centered(x, head)
    variance, mean = torch.var_mean(distance, unbiased=False)
    variance, mean = float(variance), float(mean)
    if not math.isfinite(variance) or not math.isfinite(mean):
        raise ValueError("Bayesian Deep SVDD produced non-finite training scores.")
    relative_std = math.sqrt(variance) / mean if mean > 0 else 0.0
    first = torch.as_tensor(head["first_mean"], device=x.device)
    second = torch.as_tensor(head["second_mean"], device=x.device)
    embeddings = torch.cat([embed(batch, first, second) for batch in x.split(BATCH_SIZE)])
    embedding_variance = float(embeddings.var(dim=0, unbiased=False).mean())
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
