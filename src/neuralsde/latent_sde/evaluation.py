"""Streaming Monte Carlo validation, independent of training RNG and writer."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .diagnostics import DynamicsAggregate, HistogramSamples
from .metrics import MetricMeans, loss_values, mc_standard_error
from .model import NeuralLatentSDE


@dataclass
class ValidationResult:
    metrics: dict[str, float]
    initial_metrics: dict[str, float]
    histograms: dict[str, Tensor]
    dynamics: DynamicsAggregate
    examples: Tensor | None
    reconstructions: Tensor | None
    time_mse: Tensor
    time_std: Tensor
    decoder_residual_rms: float | None


def trajectories_from_batch(batch: Tensor | list[Tensor] | tuple[Tensor, ...]) -> Tensor:
    if isinstance(batch, Tensor):
        return batch
    if len(batch) != 1:
        raise ValueError("dataset items must contain one trajectory tensor")
    return batch[0]


def validate_trajectories(trajectories: Tensor, time_points: int) -> None:
    if trajectories.ndim != 3 or tuple(trajectories.shape[1:]) != (time_points, 1):
        raise ValueError(f"trajectories must have shape (paths, {time_points}, 1), got {tuple(trajectories.shape)}")


@torch.no_grad()
def validate(
    model: NeuralLatentSDE,
    validation_loader: DataLoader,
    times: Tensor,
    *,
    kl_weight: float,
    device: torch.device,
    samples: int = 32,
    seed: int = 7,
    probe_paths: int = 0,
    plot_paths: int = 0,
) -> ValidationResult:
    """Average independent draws per example; reset seeds between epochs."""
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    model.eval()
    generators = [torch.Generator(device=device).manual_seed(seed + 1000 + s) for s in range(samples)]
    metrics, histograms, dynamics = MetricMeans(), HistogramSamples(), DynamicsAggregate()
    replicate_sums = torch.zeros(samples, 3, dtype=torch.float64)
    mc_names = ("loss", "nll", "kl_path")
    total_paths = len(validation_loader.dataset)
    total_probe = min(probe_paths, total_paths)
    initial_means, initial_log_stds = [], []
    examples, reconstructions = [], []
    time_mse = torch.zeros(len(times), dtype=torch.float64)
    time_std = torch.zeros_like(time_mse)
    mean_error_sum, residual_sum, residual_count, seen = 0.0, 0.0, 0, 0
    for batch in validation_loader:
        trajectories = trajectories_from_batch(batch).to(device)
        validate_trajectories(trajectories, len(times))
        size = len(trajectories)
        probe = min(size, max(0, probe_paths - seen))
        plotted = min(size, max(0, plot_paths - seen))
        running_mean = torch.zeros_like(trajectories, dtype=torch.float64)
        running_m2 = torch.zeros_like(running_mean)
        plotted_draws = []
        for sample, generator in enumerate(generators):
            output = model.loss(trajectories, times, kl_weight=kl_weight, generator=generator, diagnostic_paths=probe)
            values = loss_values(output, trajectories, kl_weight)
            batch_sums = metrics.add(values)
            replicate_sums[sample] += torch.tensor(
                [batch_sums[key] for key in mc_names], dtype=torch.float64,
            )
            reconstructed = output.reconstruction.detach()
            delta = reconstructed.double() - running_mean
            running_mean += delta / (sample + 1)
            running_m2 += delta * (reconstructed.double() - running_mean)
            time_mse += (reconstructed - trajectories).double().square().sum(dim=(0, 2)).cpu()
            if sample == 0:
                initial_means.append(output.posterior.initial_mean.cpu())
                initial_log_stds.append(output.posterior.initial_log_std.cpu())
            if probe:
                diagnostics = output.posterior.diagnostics
                assert diagnostics is not None
                dynamics.add(diagnostics)
                for name, tensor in diagnostics.grid_values.items():
                    histograms.add(f"posterior/{name}", tensor, total_probe * samples * len(times))
                tensors = {"latent": output.posterior.paths[:probe], "decoded": reconstructed[:probe],
                           "reconstruction_error": reconstructed[:probe] - trajectories[:probe]}
                for name, tensor in tensors.items():
                    histograms.add(f"posterior/{name}", tensor, total_probe * samples * len(times))
                residual_sum += (reconstructed[:probe] - output.posterior.paths[:probe]).double().square().sum().item()
                residual_count += probe * len(times)
            if plotted:
                plotted_draws.append(reconstructed[:plotted].cpu().clone())
        mean_error_sum += (running_mean - trajectories).square().sum().item()
        # Population spread of the finite set of posterior draws, as in its quantile plots.
        time_std += (running_m2 / samples).clamp_min(0).sqrt().sum(dim=(0, 2)).cpu()
        if plotted:
            examples.append(trajectories[:plotted].cpu())
            reconstructions.append(torch.stack(plotted_draws))
        seen += size
    if not seen:
        raise ValueError("validation_loader must yield at least one trajectory")
    result = metrics.result()
    result["posterior_mean_mse"] = mean_error_sum / (seen * len(times))
    result["posterior_mean_rmse"] = math.sqrt(result["posterior_mean_mse"])
    result["posterior_std_mean"] = (time_std / seen).mean().item()
    if samples > 1:
        for key, value in zip(mc_names, mc_standard_error(replicate_sums / seen)):
            result[f"{key}_mc_se"] = value.item()
    means, logs = torch.cat(initial_means).double(), torch.cat(initial_log_stds).double()
    stds = logs.clamp(-10, 5).exp()
    initial = {
        "mean_mean": means.mean().item(), "mean_std": means.std(correction=0).item(),
        "std_mean": stds.mean().item(), "std_min": stds.min().item(), "std_max": stds.max().item(),
        "lower_clamp_fraction": (logs < -10).double().mean().item(),
        "upper_clamp_fraction": (logs > 5).double().mean().item(),
    }
    if probe_paths:
        histograms.add("initial/posterior_mean", means, means.numel())
        histograms.add("initial/posterior_std", stds, stds.numel())
    return ValidationResult(
        result, initial, histograms.tensors(), dynamics,
        torch.cat(examples) if examples else None,
        torch.cat(reconstructions, dim=1) if reconstructions else None,
        time_mse / (seen * samples), time_std / seen,
        math.sqrt(residual_sum / residual_count) if residual_count else None,
    )
