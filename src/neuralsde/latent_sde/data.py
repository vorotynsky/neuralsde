from pathlib import Path
from typing import Any

import torch
from torch.utils.data import TensorDataset

from neuralsde.synthetic_data import PROCESS_TYPES, validate_time_grid

from .config import DatasetConfig


class TrajectoryDataset(TensorDataset):
    """Clean trajectories on one time grid shared by every process."""

    def __init__(
        self,
        trajectories: torch.Tensor,
        shared_time_grid: torch.Tensor,
        *,
        generation_config: dict[str, Any] | None = None,
    ) -> None:
        if trajectories.ndim != 3 or trajectories.shape[-1] != 1:
            raise ValueError("trajectories must have shape (paths, times, 1)")

        validate_time_grid(shared_time_grid, name="shared_time_grid")
        if len(shared_time_grid) != trajectories.shape[1]:
            raise ValueError("shared_time_grid must match the trajectory length")

        if not torch.isfinite(trajectories).all():
            raise ValueError("trajectories must be finite")

        trajectories = trajectories.detach().cpu()
        shared_time_grid = shared_time_grid.detach().cpu()

        super().__init__(trajectories)

        self.shared_time_grid = shared_time_grid
        self.generation_config = generation_config

    def save(self, path: str | Path) -> None:
        """Save tensors and generation metadata as a portable PyTorch file."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "trajectories": self.tensors[0],
                "shared_time_grid": self.shared_time_grid,
                "generation_config": self.generation_config,
            },
            output,
        )

    @classmethod
    def load(cls, path: str | Path) -> TrajectoryDataset:
        """Load and validate a dataset created by :meth:`save`."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("dataset file must contain a mapping")

        required = {"trajectories", "shared_time_grid"}
        missing = required - payload.keys()
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"dataset file is missing: {names}")

        return cls(
            payload["trajectories"],
            payload["shared_time_grid"],
            generation_config=payload.get("generation_config"),
        )


def build_dataset(
    config: DatasetConfig, *, device: str | torch.device = "cpu"
) -> TrajectoryDataset:
    """Sample every configured process on one shared grid."""
    device = torch.device(device)
    shared_time_grid = torch.linspace(
        0.0, config.horizon, config.steps + 1, device=device
    )
    generator = torch.Generator(device=device).manual_seed(config.seed)
    paths = [
        PROCESS_TYPES[sampler.type].build(sampler.params).sample(
            sampler.paths, shared_time_grid, generator=generator
        )
        for sampler in config.samplers
    ]
    observations = torch.cat(paths)

    return TrajectoryDataset(
        observations,
        shared_time_grid,
        generation_config=config.to_dict(),
    )


def load_dataset(path: str | Path) -> TrajectoryDataset:
    """Load a generated trajectory dataset from disk."""
    return TrajectoryDataset.load(path)
