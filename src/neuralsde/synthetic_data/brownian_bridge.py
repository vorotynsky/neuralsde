from dataclasses import dataclass, field

import torch
from torch import Tensor

from ._validation import validate_sampling_inputs, warn_nonzero_initial


@dataclass(frozen=True)
class BrownianBridgeParameters:
    r"""Parameters of a Brownian bridge from ``x0`` to ``x1``.

    The underlying Brownian motion has volatility ``sigma``. The bridge's
    terminal time is the last value in the time grid passed to ``sample``.
    """

    x0: float = 0.0
    x1: float = 0.0
    sigma: float = 1.0

    def validate(self) -> None:
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")

        warn_nonzero_initial(self.x0)


@dataclass(frozen=True)
class BrownianBridgeProcess:
    params: BrownianBridgeParameters = field(default_factory=BrownianBridgeParameters)

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
        endpoint = torch.as_tensor(
            self.params.x1, device=times.device, dtype=times.dtype
        )
        sigma = torch.as_tensor(
            self.params.sigma, device=times.device, dtype=times.dtype
        )
        terminal_time = times[-1]

        for step, (current_time, next_time) in enumerate(zip(times[:-1], times[1:])):
            remaining = terminal_time - current_time
            dt = next_time - current_time
            mean = latent[:, step, 0] + (endpoint - latent[:, step, 0]) * dt / remaining
            variance = sigma.square() * dt * (terminal_time - next_time) / remaining
            noise = torch.randn(
                number_of_paths,
                device=times.device,
                dtype=times.dtype,
                generator=generator,
            )
            latent[:, step + 1, 0] = mean + torch.sqrt(variance) * noise

        return latent


def sample_brownian_bridge(
    number_of_paths: int,
    times: Tensor,
    params: BrownianBridgeParameters = BrownianBridgeParameters(),
    *,
    generator: torch.Generator,
) -> Tensor:
    return BrownianBridgeProcess(params).sample(
        number_of_paths, times, generator=generator
    )
