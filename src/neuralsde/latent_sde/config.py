from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import math
import yaml

from neuralsde.synthetic_data import (
    BrownianBridgeParameters,
    CIRParameters,
    OUParameters,
    PROCESS_TYPES,
)


ProcessParameters = OUParameters | CIRParameters | BrownianBridgeParameters


@dataclass(frozen=True)
class SamplerConfig:
    """Configuration for one group of synthetic trajectories."""

    type: str
    paths: int
    params: ProcessParameters | dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        process_type = self.type.lower()
        try:
            expected_type = PROCESS_TYPES[process_type].parameter_type
        except KeyError:
            raise ValueError(f"Unknown sampler type: {self.type}")
        if self.paths <= 0:
            raise ValueError("sampler paths must be positive")

        params = self.params
        if isinstance(params, dict):
            try:
                params = expected_type(**params)
            except TypeError as error:
                raise ValueError(f"Invalid parameters for {process_type}: {error}") from error
        elif not isinstance(params, expected_type):
            raise ValueError(f"Parameters for {process_type} must be {expected_type.__name__}")
        params.validate()
        object.__setattr__(self, "type", process_type)
        object.__setattr__(self, "params", params)


def default_samplers() -> tuple[SamplerConfig, ...]:
    return (
        SamplerConfig("ou", 512, {"theta": 0.8, "mu": 1.0, "sigma": 0.45}),
        SamplerConfig("ou", 256, {"theta": 1.4, "mu": 2.0, "sigma": 0.25}),
        SamplerConfig("cir", 256, {"kappa": 1.2, "theta": 0.8, "sigma": 0.35}),
        SamplerConfig("brownian_bridge", 256, {"x1": 0.0, "sigma": 0.6}),
    )


def _load_yaml(path: str | Path, config_name: str) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open() as config_file:
        values = yaml.safe_load(config_file)
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise ValueError(f"{config_name} config must be a mapping: {config_path}")
    return values


def _reject_unknown_keys(values: dict[str, Any], config_type: type) -> None:
    known_keys = {config_field.name for config_field in fields(config_type)}
    unknown_keys = set(values) - known_keys
    if unknown_keys:
        names = ", ".join(sorted(unknown_keys))
        raise ValueError(f"Unknown {config_type.__name__} key(s): {names}")


def _positive_integer(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class TensorBoardConfig:
    """Recording cadence and bounded diagnostic sample sizes."""

    enabled: bool = True
    batch_every: int = 10
    diagnostics_every: int = 25
    prior_samples: int = 1024
    posterior_probe_paths: int = 64
    plot_paths: int = 8

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("tensorboard.enabled must be a boolean")
        for name in ("batch_every", "diagnostics_every", "prior_samples", "posterior_probe_paths", "plot_paths"):
            _positive_integer(f"tensorboard.{name}", getattr(self, name))


@dataclass(frozen=True)
class ExperimentConfig:
    """Parameters that define training, independent of dataset generation."""

    epochs: int = 900
    batch_size: int = 128
    sample_path: int = 4096
    validation_size: int | float = 0.2
    validation_samples: int = 32
    learning_rate: float = 3e-3
    solver_dt: float = 0.01
    observation_std: float = 0.1
    initial_prior_std: float = 0.1
    kl_weight: float = 1.0
    log_every: int = 100
    seed: int = 7
    tensorboard: TensorBoardConfig = field(default_factory=TensorBoardConfig)

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "sample_path", "validation_samples", "log_every"):
            _positive_integer(name, getattr(self, name))
        if not isinstance(self.tensorboard, TensorBoardConfig):
            raise ValueError("tensorboard must be a TensorBoardConfig")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.solver_dt) or self.solver_dt <= 0:
            raise ValueError("solver_dt must be finite and positive")
        if not math.isfinite(self.observation_std) or self.observation_std <= 0:
            raise ValueError("observation_std must be finite and positive")
        if not math.isfinite(self.initial_prior_std) or self.initial_prior_std <= 0:
            raise ValueError("initial_prior_std must be finite and positive")
        if not math.isfinite(self.kl_weight) or self.kl_weight < 0:
            raise ValueError("kl_weight must be finite and non-negative")
        size = self.validation_size
        if isinstance(size, bool) or not isinstance(size, (int, float)):
            raise ValueError("validation_size must be an integer count or a float fraction")
        if isinstance(size, int) and size < 1:
            raise ValueError("integer validation_size must be positive")
        if isinstance(size, float) and (not math.isfinite(size) or not 0.0 < size < 1.0):
            raise ValueError("float validation_size must be between 0 and 1")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load a training config and reject dataset-generation keys."""
        values = _load_yaml(path, "Training")
        _reject_unknown_keys(values, cls)
        if "tensorboard" in values:
            if not isinstance(values["tensorboard"], dict):
                raise ValueError("tensorboard must be a mapping")
            _reject_unknown_keys(values["tensorboard"], TensorBoardConfig)
            values["tensorboard"] = TensorBoardConfig(**values["tensorboard"])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetConfig:
    """Parameters used to generate one reusable synthetic dataset."""

    steps: int = 100
    horizon: float = 2.0
    seed: int = 8
    samplers: tuple[SamplerConfig, ...] = field(default_factory=default_samplers)

    def __post_init__(self) -> None:
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("steps must be a positive integer")
        if not isinstance(self.horizon, (int, float)) or not math.isfinite(self.horizon):
            raise ValueError("horizon must be finite")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not self.samplers:
            raise ValueError("samplers must be a non-empty tuple")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DatasetConfig":
        """Load a dataset config and reject unknown keys."""
        values = _load_yaml(path, "Dataset")
        _reject_unknown_keys(values, cls)
        values = dict(values)
        if "samplers" in values:
            samplers = values["samplers"]
            if not isinstance(samplers, list) or not samplers:
                raise ValueError("samplers must be a non-empty list")
            if not all(isinstance(sampler, dict) for sampler in samplers):
                raise ValueError("each sampler must be a mapping")
            try:
                values["samplers"] = tuple(SamplerConfig(**sampler) for sampler in samplers)
            except TypeError as error:
                raise ValueError(f"Invalid sampler configuration: {error}") from error
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly representation."""
        values = asdict(self)
        values["samplers"] = list(values["samplers"])
        return values
