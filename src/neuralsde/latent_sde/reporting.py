"""TensorBoard lifecycle, prior probes, and headless diagnostic figures."""

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
from uuid import uuid4

import torch
from torch import Tensor

from .config import ExperimentConfig
from .diagnostics import DynamicsAggregate, HistogramSamples, thin
from .evaluation import ValidationResult
from .metrics import covariance, generation_metrics
from .model import NeuralLatentSDE


@dataclass
class PriorDiagnostics:
    metrics: dict[str, float]
    decoded: Tensor
    dynamics: DynamicsAggregate
    histograms: dict[str, Tensor]
    decoder_residual_rms: float


@torch.no_grad()
def prior_diagnostics(
    model: NeuralLatentSDE, times: Tensor, reference: Tensor, *, samples: int,
    batch_size: int, seed: int,
) -> PriorDiagnostics:
    generator = torch.Generator(device=times.device).manual_seed(seed + 2000)
    dynamics, histograms = DynamicsAggregate(), HistogramSamples()
    decoded, residual_sum = [], 0.0
    for start in range(0, samples, batch_size):
        count = min(batch_size, samples - start)
        integrated = model._integrate(count, times, generator=generator, diagnostic_paths=count)
        reconstruction = model.decode(integrated.paths)
        if not torch.isfinite(reconstruction).all():
            raise FloatingPointError("non-finite prior trajectories")
        decoded.append(reconstruction.cpu())
        assert integrated.diagnostics is not None
        dynamics.add(integrated.diagnostics)
        for name, values in {**integrated.diagnostics.grid_values, "latent": integrated.paths, "decoded": reconstruction}.items():
            histograms.add(f"prior/{name}", values, samples * len(times))
        residual_sum += (reconstruction - integrated.paths).double().square().sum().item()
    paths = torch.cat(decoded)
    return PriorDiagnostics(generation_metrics(paths, reference), paths, dynamics, histograms.tensors(),
                            math.sqrt(residual_sum / (samples * len(times))))


class TensorBoardLogger:
    def __init__(self, output: str | Path, config: ExperimentConfig, metadata: dict) -> None:
        self.writer = None
        self.run_dir: Path | None = None
        self.metadata = {**metadata, "config": config.to_dict(), "status": "running"}
        if not config.tensorboard.enabled:
            return
        from torch.utils.tensorboard import SummaryWriter

        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid4().hex[:8]
        self.run_dir = Path(output) / "tensorboard" / name
        self.writer = SummaryWriter(log_dir=str(self.run_dir))
        try:
            self.metadata["versions"] = {
                package: importlib.metadata.version(package)
                for package in ("torch", "tensorboard", "matplotlib", "neuralsde")
            }
            self._save_metadata()
            self.writer.add_text("run/metadata", "```json\n" + json.dumps(self.metadata, indent=2) + "\n```", 0)
            print(f"TensorBoard: {self.run_dir}")
        except BaseException:
            self.writer.close()
            raise

    def _save_metadata(self) -> None:
        if self.run_dir is not None:
            (self.run_dir / "metadata.json").write_text(json.dumps(self.metadata, indent=2, allow_nan=False) + "\n")

    def scalars(self, prefix: str, values: dict[str, float], step: int) -> None:
        if self.writer is not None:
            for name, value in values.items():
                if not math.isfinite(value):
                    raise FloatingPointError(f"non-finite logged metric: {prefix}/{name}")
                self.writer.add_scalar(f"{prefix}/{name}", value, step)

    def histograms(self, values: dict[str, Tensor], epoch: int) -> None:
        if self.writer is not None:
            for name, value in values.items():
                sampled = thin(value)
                if not torch.isfinite(sampled).all():
                    raise FloatingPointError(f"non-finite histogram: {name}")
                self.writer.add_histogram(f"histograms/{name}", sampled, epoch)

    def figures(self, times: Tensor, validation: ValidationResult, prior: PriorDiagnostics,
                reference: Tensor, epoch: int, plot_paths: int) -> None:
        if self.writer is None:
            return
        for tag, figure in diagnostic_figures(times, validation, prior, reference, plot_paths):
            try:
                self.writer.add_figure(f"figures/{tag}", figure, epoch, close=True)
            finally:
                figure.clear()

    def flush(self) -> None:
        if self.writer is not None:
            self.writer.flush()

    def close(self, *, status: str, epoch: int, step: int, error: str | None = None) -> None:
        if self.writer is None:
            return
        try:
            self.metadata.update(status=status, final_epoch=epoch, final_step=step)
            if error is not None:
                self.metadata["error"] = error
            self._save_metadata()
            self.writer.add_text("run/status", json.dumps({"status": status, "error": error}), step)
            self.writer.flush()
        finally:
            self.writer.close()


def diagnostic_figures(times: Tensor, validation: ValidationResult, prior: PriorDiagnostics,
                       reference: Tensor, plot_paths: int):
    # Figure objects and Agg avoid GUI backends and pyplot's global registry.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    def figure(rows=1, cols=1, *, width=11, height=3.5):
        fig = Figure(figsize=(width, height * rows), layout="constrained")
        FigureCanvasAgg(fig)
        return fig, fig.subplots(rows, cols, squeeze=False)

    def finish_axis(ax, title, ylabel="Value"):
        ax.set(title=title, xlabel="Time", ylabel=ylabel)
        ax.grid(alpha=0.2)

    t = times.detach().cpu().numpy()
    actual, generated = reference.cpu().squeeze(-1), prior.decoded.squeeze(-1)
    if validation.examples is not None and validation.reconstructions is not None:
        examples = validation.examples.squeeze(-1)
        draws = validation.reconstructions.squeeze(-1)
        count = len(examples)
        fig, axes = figure(math.ceil(count / 2), 2, height=2.5)
        low, high = draws.quantile(0.05, dim=0), draws.quantile(0.95, dim=0)
        for index, ax in enumerate(axes.flat):
            if index >= count:
                ax.set_visible(False)
                continue
            ax.fill_between(t, low[index], high[index], color="tab:blue", alpha=0.2, label="Posterior 5–95%")
            ax.plot(t, draws[:3, index].T, color="tab:blue", alpha=0.2, linewidth=0.8)
            ax.plot(t, draws[:, index].mean(0), color="tab:blue", label="Posterior mean")
            ax.plot(t, examples[index], color="black", linewidth=1.3, label="Observed")
            finish_axis(ax, f"Validation example {index + 1}")
            if index == 0:
                ax.legend(fontsize=8)
        yield "posterior_reconstruction", fig

    fig, axes = figure(1, 2)
    axes[0, 0].plot(t, validation.time_mse)
    axes[0, 1].plot(t, validation.time_std)
    finish_axis(axes[0, 0], "Posterior sample reconstruction error", "MSE")
    finish_axis(axes[0, 1], "Posterior decoded spread", "Standard deviation")
    yield "reconstruction_error", fig

    fig, axes = figure()
    axes[0, 0].plot(t, generated[:plot_paths].T, alpha=0.7)
    finish_axis(axes[0, 0], "Independent decoded prior paths")
    yield "prior_paths", fig

    fig, axes = figure()
    ax = axes[0, 0]
    for label, data, color in (("Validation", actual, "black"), ("Prior", generated, "tab:blue")):
        ax.fill_between(t, data.quantile(0.05, dim=0), data.quantile(0.95, dim=0), color=color, alpha=0.15)
        ax.plot(t, data.mean(0), color=color, label=f"{label}: mean and 5–95%")
    finish_axis(ax, "Decoded prior and empirical marginals")
    ax.legend()
    yield "prior_marginals", fig

    indices = list(dict.fromkeys((0, len(t) // 2, len(t) - 1)))
    fig, axes = figure(1, len(indices))
    for index, ax in zip(indices, axes.flat):
        # ECDF remains meaningful for deterministic initial states and singleton datasets.
        combined = torch.cat((actual[:, index], generated[:, index]))
        margin = max((combined.max() - combined.min()).item() * 0.05, 1e-3)
        left, right = combined.min() - margin, combined.max() + margin
        for label, data, color in (("Validation", actual, "black"), ("Prior", generated, "tab:blue")):
            values = data[:, index].sort().values
            support = torch.cat((left[None], values, right[None]))
            cdf = torch.cat((torch.zeros(1), torch.arange(1, len(values) + 1) / len(values), torch.ones(1)))
            ax.step(support.numpy(), cdf.numpy(), where="post", color=color, label=label)
        ax.set(title=f"Time {t[index]:.3g}", xlabel="Value", ylabel="Empirical CDF")
        ax.legend()
        ax.grid(alpha=0.2)
    yield "marginal_distributions", fig

    if min(len(actual), len(generated)) >= 2:
        observed_cov, generated_cov = covariance(actual.double()), covariance(generated.double())
        limit = max(observed_cov.abs().max().item(), generated_cov.abs().max().item(), 1e-12)
        fig, axes = figure(1, 3, width=14, height=4)
        for ax, data, title in zip(axes.flat, (observed_cov, generated_cov, generated_cov - observed_cov),
                                   ("Validation covariance", "Prior covariance", "Prior minus validation")):
            scale = max(data.abs().max().item(), 1e-12) if title.startswith("Prior minus") else limit
            mesh = ax.pcolormesh(t, t, data.numpy(), shading="nearest", cmap="RdBu_r", vmin=-scale, vmax=scale)
            ax.set(title=title, xlabel="Time", ylabel="Time")
            fig.colorbar(mesh, ax=ax, shrink=0.8)
        yield "temporal_covariance", fig

    midpoints = (t[:-1] + t[1:]) / 2
    posterior_curves = validation.dynamics.curves(times)
    prior_curves = prior.dynamics.curves(times)
    fig, axes = figure(1, 2)
    for key in ("prior_drift_rms", "posterior_drift_rms", "drift_gap_rms"):
        axes[0, 0].plot(midpoints, posterior_curves[key], label=key.replace("_", " "))
    axes[0, 0].plot(midpoints, prior_curves["prior_drift_rms"], linestyle="--", label="drift on prior")
    for label, curves in (("Posterior", posterior_curves), ("Prior", prior_curves)):
        axes[0, 1].plot(midpoints, curves["diffusion"], label=label)
    finish_axis(axes[0, 0], "Time-weighted drift RMS", "RMS")
    finish_axis(axes[0, 1], "Time-weighted diffusion", "Mean diffusion")
    for ax in axes.flat:
        ax.legend(fontsize=8)
    yield "dynamics", fig

    fig, axes = figure()
    axes[0, 0].stairs(posterior_curves["path_kl_density"].numpy(), t)
    finish_axis(axes[0, 0], "Posterior path KL per unit time", "KL density")
    yield "path_kl_density", fig
