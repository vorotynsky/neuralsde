from dataclasses import dataclass, field

import torch
from torch import Tensor

from ._validation import validate_sampling_inputs, warn_nonzero_initial


@dataclass(frozen=True)
class CIRParameters:
    r"""Parameters of \(dX_t=\kappa(\vartheta-X_t)dt+\sigma\sqrt{X_t}dW_t\)."""

    kappa: float = 1.2
    theta: float = 0.8
    sigma: float = 0.35
    x0: float = 0.0

    def validate(self) -> None:
        if self.kappa <= 0:
            raise ValueError("kappa must be positive")
        if self.theta <= 0:
            raise ValueError("theta must be positive")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.x0 < 0:
            raise ValueError("x0 must be non-negative")

        warn_nonzero_initial(self.x0)


@dataclass(frozen=True)
class CIRProcess:
    params: CIRParameters = field(default_factory=CIRParameters)

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
        kappa = torch.as_tensor(self.params.kappa, device=times.device, dtype=times.dtype)
        theta = torch.as_tensor(self.params.theta, device=times.device, dtype=times.dtype)
        sigma = torch.as_tensor(self.params.sigma, device=times.device, dtype=times.dtype)
        degrees_of_freedom = 4.0 * kappa * theta / sigma.square()

        for step, dt in enumerate(times[1:] - times[:-1]):
            decay = torch.exp(-kappa * dt)
            scale = sigma.square() * (1.0 - decay) / (4.0 * kappa)
            noncentrality = decay * latent[:, step, 0] / scale
            poisson_counts = torch.poisson(noncentrality / 2.0, generator=generator)
            gamma_shape = degrees_of_freedom / 2.0 + poisson_counts
            chi_squared = 2.0 * torch._standard_gamma(gamma_shape, generator=generator)
            latent[:, step + 1, 0] = scale * chi_squared
        return latent


def sample_cir(
    number_of_paths: int,
    times: Tensor,
    params: CIRParameters = CIRParameters(),
    *,
    generator: torch.Generator,
) -> Tensor:
    return CIRProcess(params).sample(number_of_paths, times, generator=generator)
