"""Detached, time-weighted solver diagnostics and bounded histogram samples."""

from dataclasses import dataclass, field

import torch
from torch import Tensor


HISTOGRAM_LIMIT = 16_384


def thin(values: Tensor, limit: int = HISTOGRAM_LIMIT) -> Tensor:
    values = values.detach().flatten().cpu()
    if values.numel() > limit:
        values = values[torch.linspace(0, values.numel() - 1, limit).long()]
    return values


class HistogramSamples:
    """Select evenly spaced global indices from a stream of known length."""

    def __init__(self) -> None:
        self._offsets: dict[str, int] = {}
        self._values: dict[str, list[Tensor]] = {}

    def add(self, name: str, values: Tensor, total: int) -> None:
        values = values.detach().flatten()
        offset = self._offsets.get(name, 0)
        indices = torch.linspace(0, total - 1, min(total, HISTOGRAM_LIMIT)).long()
        selected = indices[(indices >= offset) & (indices < offset + values.numel())] - offset
        if len(selected):
            self._values.setdefault(name, []).append(values[selected.to(values.device)].cpu())
        self._offsets[name] = offset + values.numel()

    def tensors(self) -> dict[str, Tensor]:
        return {name: torch.cat(parts) for name, parts in self._values.items()}


@dataclass
class IntegrationDiagnostics:
    paths: int
    integrals: dict[str, Tensor]
    diffusion_min: Tensor
    diffusion_max: Tensor
    grid_values: dict[str, Tensor]


class SolverDiagnostics:
    """Collect only the requested leading paths without retaining autograd."""

    def __init__(self, times: Tensor, paths: int) -> None:
        self.paths = paths
        self.integrals: dict[str, Tensor] = {}
        self.template = times.new_zeros(len(times) - 1)
        self.diffusion_min = times.new_tensor(float("inf"))
        self.diffusion_max = times.new_tensor(float("-inf"))
        self.grid: dict[str, list[Tensor]] = {}

    def fields(self, drift: Tensor, diffusion: Tensor, posterior: Tensor | None) -> dict[str, Tensor]:
        f, g = drift[:self.paths].detach(), diffusion[:self.paths].detach()
        result = {"prior_drift": f, "diffusion": g}
        if posterior is not None:
            h = posterior[:self.paths].detach()
            result.update(posterior_drift=h, drift_gap=h - f, control=(h - f) / g)
        return result

    @torch.no_grad()
    def add(self, index: int, dt: Tensor, fields: dict[str, Tensor], floor: float) -> None:
        for name, value in fields.items():
            key = "diffusion" if name == "diffusion" else f"{name}_square"
            measure = value if name == "diffusion" else value.square()
            self.integrals.setdefault(key, self.template.clone())[index] += measure.sum() * dt
        g = fields["diffusion"]
        self.integrals.setdefault("near_floor", self.template.clone())[index] += (g < 10 * floor).sum() * dt
        self.diffusion_min = torch.minimum(self.diffusion_min, g.min())
        self.diffusion_max = torch.maximum(self.diffusion_max, g.max())

    def snapshot(self, fields: dict[str, Tensor]) -> None:
        for name, value in fields.items():
            self.grid.setdefault(name, []).append(value)

    def finish(self) -> IntegrationDiagnostics:
        return IntegrationDiagnostics(
            self.paths,
            {key: value.cpu() for key, value in self.integrals.items()},
            self.diffusion_min.cpu(), self.diffusion_max.cpu(),
            {key: torch.stack(values, dim=1).cpu() for key, values in self.grid.items()},
        )


@dataclass
class DynamicsAggregate:
    paths: int = 0
    integrals: dict[str, Tensor] = field(default_factory=dict)
    diffusion_min: float = float("inf")
    diffusion_max: float = float("-inf")

    def add(self, result: IntegrationDiagnostics) -> None:
        self.paths += result.paths
        for key, value in result.integrals.items():
            self.integrals[key] = self.integrals.get(key, torch.zeros_like(value)) + value
        self.diffusion_min = min(self.diffusion_min, result.diffusion_min.item())
        self.diffusion_max = max(self.diffusion_max, result.diffusion_max.item())

    def curves(self, times: Tensor) -> dict[str, Tensor]:
        denominator = self.paths * times.detach().cpu().diff()
        result = {}
        for name, values in self.integrals.items():
            mean = values / denominator
            key = name.removesuffix("_square") + "_rms" if name.endswith("_square") else name
            result[key] = mean.clamp_min(0).sqrt() if name.endswith("_square") else mean
        if "control_square" in self.integrals:
            result["path_kl_density"] = 0.5 * self.integrals["control_square"] / denominator
        return result

    def metrics(self, times: Tensor) -> dict[str, float]:
        denominator = self.paths * (times[-1] - times[0]).item()
        result = {"diffusion_min": self.diffusion_min, "diffusion_max": self.diffusion_max}
        for name, values in self.integrals.items():
            mean = values.sum().item() / denominator
            if name.endswith("_square"):
                result[name.removesuffix("_square") + "_rms"] = max(mean, 0.0) ** 0.5
            else:
                result["diffusion_mean" if name == "diffusion" else "diffusion_near_floor_fraction"] = mean
        return result
