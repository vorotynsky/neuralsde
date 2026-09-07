import torch
from torch import Tensor
import warnings


def validate_time_grid(time_grid: Tensor, *, name: str = "time_grid") -> None:
    """Validate a finite, strictly increasing, floating-point time grid."""
    if not isinstance(time_grid, Tensor) or time_grid.ndim != 1 or len(time_grid) < 2:
        raise ValueError(
            f"{name} must be a one-dimensional tensor with at least two values"
        )
    if not time_grid.is_floating_point():
        raise ValueError(f"{name} must have a floating-point dtype")
    if not torch.isfinite(time_grid).all():
        raise ValueError(f"{name} must contain only finite values")
    if not torch.all(time_grid[1:] > time_grid[:-1]):
        raise ValueError(f"{name} must be strictly increasing")


def validate_sampling_inputs(number_of_paths: int, times: Tensor) -> None:
    if not isinstance(number_of_paths, int) or number_of_paths <= 0:
        raise ValueError("number_of_paths must be a positive integer")
    validate_time_grid(times, name="times")


def warn_nonzero_initial(x0):
    if x0 != 0.0:
        warnings.warn("The model is not support processes started from arbitary points. Initial value have to be zero.")
