"""Sample-weighted objectives and empirical generative quality metrics."""

import math

import torch
from torch import Tensor

from .model import LossOutput


def loss_values(output: LossOutput, observations: Tensor, kl_weight: float) -> dict[str, Tensor]:
    """One value per path; components retain their normalization explicitly."""
    points = observations.shape[1]
    nll = output.nll_per_path.detach() / points
    initial = output.posterior.initial_kl.detach() / points
    path = output.posterior.path_kl.detach() / points
    error = output.reconstruction.detach() - observations
    return {
        "loss": nll + kl_weight * (initial + path),
        "negative_elbo": nll + initial + path,
        "nll": nll, "kl_initial": initial, "kl_path": path,
        "kl_total": initial + path, "kl_weighted": kl_weight * (initial + path),
        "kl_initial_per_path": output.posterior.initial_kl.detach(),
        "kl_path_per_path": output.posterior.path_kl.detach(),
        "reconstruction_mse": error.square().mean(dim=(1, 2)),
        "reconstruction_mae": error.abs().mean(dim=(1, 2)),
    }


class MetricMeans:
    def __init__(self) -> None:
        self.count = 0
        self.sums: dict[str, float] = {}

    def add(self, values: dict[str, Tensor]) -> dict[str, float]:
        """Accumulate detached sums with a single device-to-host transfer."""
        count = next(iter(values.values())).numel()
        self.count += count
        totals = torch.stack([value.detach().double().sum() for value in values.values()]).cpu().tolist()
        batch_sums = dict(zip(values, totals))
        for name, total in batch_sums.items():
            if not math.isfinite(total):
                raise FloatingPointError(f"non-finite metric: {name}")
            self.sums[name] = self.sums.get(name, 0.0) + total
        return batch_sums

    def result(self) -> dict[str, float]:
        if not self.count:
            raise ValueError("metrics require at least one trajectory")
        result = {name: value / self.count for name, value in self.sums.items()}
        if "reconstruction_mse" in result:
            result["reconstruction_rmse"] = math.sqrt(result["reconstruction_mse"])
        return result


def mc_standard_error(replicate_means: Tensor) -> Tensor:
    """Standard error across complete-dataset Monte Carlo replicates."""
    if len(replicate_means) < 2:
        raise ValueError("Monte Carlo standard error requires at least two samples")
    return replicate_means.std(dim=0, correction=1) / math.sqrt(len(replicate_means))


def wasserstein_1(first: Tensor, second: Tensor) -> Tensor:
    """Exact empirical 1D W1, column-wise, including unequal sample counts."""
    first, second = first.double(), second.double()
    if first.ndim == 1:
        first, second = first[:, None], second[:, None]
    a = first.T.contiguous().sort(dim=-1).values
    b = second.T.contiguous().sort(dim=-1).values
    support = torch.cat((a, b), dim=-1).sort(dim=-1).values
    left = support[:, :-1].contiguous()
    cdf_a = torch.searchsorted(a, left, right=True).double() / a.shape[-1]
    cdf_b = torch.searchsorted(b, left, right=True).double() / b.shape[-1]
    return ((cdf_a - cdf_b).abs() * support.diff(dim=-1)).sum(dim=-1)


def covariance(paths: Tensor) -> Tensor:
    centered = paths - paths.mean(dim=0)
    return centered.T @ centered / len(paths)


def generation_metrics(generated: Tensor, reference: Tensor) -> dict[str, float]:
    generated = generated.detach().cpu().double().squeeze(-1)
    reference = reference.detach().cpu().double().squeeze(-1)
    if not torch.isfinite(generated).all():
        raise FloatingPointError("non-finite prior trajectories")
    marginal = wasserstein_1(generated, reference)
    increments = wasserstein_1(generated.diff(dim=1), reference.diff(dim=1))
    result = {
        "mean_rmse": (generated.mean(0) - reference.mean(0)).square().mean().sqrt().item(),
        "marginal_w1_mean": marginal.mean().item(),
        "marginal_w1_max": marginal.max().item(),
        "terminal_w1": marginal[-1].item(),
        "increment_w1_mean": increments.mean().item(),
        "quadratic_variation_mean_error": (
            generated.diff(dim=1).square().sum(1).mean()
            - reference.diff(dim=1).square().sum(1).mean()
        ).abs().item(),
    }
    if min(len(generated), len(reference)) >= 2:
        result["std_rmse"] = (generated.std(0, correction=0) - reference.std(0, correction=0)).square().mean().sqrt().item()
        result["covariance_rmse"] = (covariance(generated) - covariance(reference)).square().mean().sqrt().item()
    return result
