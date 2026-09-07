from dataclasses import dataclass, field

import torch
from torch import Tensor

from ._validation import validate_sampling_inputs, warn_nonzero_initial


@dataclass(frozen=True)
class OUParameters:
    r"""Parameters of (dX_t = \theta(\mu-X_t)dt + \sigma dW_t)."""

    theta: float = 0.8
    mu: float = 1.0
    sigma: float = 0.45
    x0: float = 0.0

    def validate(self) -> None:
        if self.theta <= 0:
            raise ValueError("theta must be positive")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")
        
        warn_nonzero_initial(self.x0)


@dataclass(frozen=True)
class OUProcess:
    params: OUParameters = field(default_factory=OUParameters)

    def sample(
        self,
        number_of_paths: int,
        times: Tensor,
        *,
        generator: torch.Generator,
    ) -> Tensor:
        validate_sampling_inputs(number_of_paths, times)
        self.params.validate()

        latent = torch.empty(
            number_of_paths, len(times), 1, device=times.device, dtype=times.dtype
        )
        latent[:, 0, 0] = self.params.x0
        theta = torch.as_tensor(self.params.theta, device=times.device, dtype=times.dtype)
        sigma = torch.as_tensor(self.params.sigma, device=times.device, dtype=times.dtype)
        mu = torch.as_tensor(self.params.mu, device=times.device, dtype=times.dtype)
        for step, dt in enumerate(times[1:] - times[:-1]):
            decay = torch.exp(-theta * dt)
            innovation_std = sigma * torch.sqrt((1.0 - decay.square()) / (2.0 * theta))
            noise = torch.randn(
                number_of_paths, device=times.device, dtype=times.dtype, generator=generator
            )
            latent[:, step + 1, 0] = (
                mu + decay * (latent[:, step, 0] - mu) + innovation_std * noise
            )
        return latent


def sample_ou(
    number_of_paths: int,
    times: Tensor,
    params: OUParameters = OUParameters(),
    *,
    generator: torch.Generator,
) -> Tensor:
    return OUProcess(params).sample(number_of_paths, times, generator=generator)
