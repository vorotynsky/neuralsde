import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from neuralsde.synthetic_data import validate_time_grid
from .diagnostics import IntegrationDiagnostics, SolverDiagnostics


@dataclass
class IntegrationResult:
    paths: Tensor
    path_kl: Tensor
    diagnostics: IntegrationDiagnostics | None = None


@dataclass
class PosteriorSample:
    paths: Tensor
    initial_kl: Tensor
    path_kl: Tensor
    initial_mean: Tensor
    initial_log_std: Tensor
    diagnostics: IntegrationDiagnostics | None = None

    @property
    def initial_std(self) -> Tensor:
        return self.initial_log_std.clamp(-10.0, 5.0).exp()


@dataclass
class LossOutput:
    """Normalized differentiable loss, with unnormalized per-path components."""

    loss: Tensor
    nll_per_path: Tensor
    reconstruction: Tensor
    posterior: PosteriorSample


class TrajectoryEncoder(nn.Module):
    """Encode observations and their times into a smoothing context."""

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.context_size = 2 * width
        self.sequence = nn.GRU(
            input_size=2,
            hidden_size=width,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, observations: Tensor, times: Tensor) -> Tensor:
        time_grid = times.view(1, -1, 1).expand_as(observations)
        features, _ = self.sequence(torch.cat((observations, time_grid), dim=-1))
        return features


class PosteriorDrift(nn.Module):
    r"""An independent smoothing drift \(h_\phi(t,z,c)\)."""

    def __init__(self, context_size: int, width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2 + context_size, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        )

    def forward(self, time: Tensor, state: Tensor, context: Tensor) -> Tensor:
        return self.network(torch.cat((time, state, context), dim=-1))


class DriftNetwork(nn.Module):
    r"""An unconstrained neural approximation to \(f_\theta(t,z)\)."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1)
        )

    def forward(self, time: Tensor, state: Tensor) -> Tensor:
        return self.network(torch.cat((time, state), dim=-1))


class DiffusionNetwork(nn.Module):
    r"""A neural approximation to \(g_\theta(t,z)>0\)."""

    def __init__(self, width: int, numerical_floor: float = 1e-4) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1)
        )
        self.numerical_floor = numerical_floor

    def forward(self, time: Tensor, state: Tensor) -> Tensor:
        raw_scale = self.network(torch.cat((time, state), dim=-1))
        return torch.nn.functional.softplus(raw_scale) + self.numerical_floor


class DecoderNetwork(nn.Module):
    """A nonlinear residual map from latent state to trajectory value."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        )
        # Keep the original identity behavior at initialization while allowing
        # training to learn nonlinear corrections to the decoded trajectory.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, state: Tensor) -> Tensor:
        return state + self.network(state)


def expand_time(time: Tensor, state: Tensor) -> Tensor:
    """Broadcast scalar or batched times to a one-dimensional state tensor."""
    time = torch.as_tensor(time, dtype=state.dtype, device=state.device)
    if time.ndim == 0:
        return time.reshape(1, 1).expand(*state.shape[:-1], 1)
    if time.shape == state.shape[:-1]:
        time = time.unsqueeze(-1)
    return torch.broadcast_to(time, state.shape)


class NeuralLatentSDE(nn.Module):
    """Variational latent SDE with Gaussian initial and observation models."""

    def __init__(
        self,
        width: int = 48,
        *,
        solver_dt: float = 0.01,
        observation_std: float = 0.1,
        initial_prior_std: float = 0.1,
    ) -> None:
        super().__init__()
        if not math.isfinite(solver_dt) or solver_dt <= 0.0:
            raise ValueError("solver_dt must be finite and positive")
        if not math.isfinite(observation_std) or observation_std <= 0.0:
            raise ValueError("observation_std must be finite and positive")
        if not math.isfinite(initial_prior_std) or initial_prior_std <= 0.0:
            raise ValueError("initial_prior_std must be finite and positive")

        self.encoder = TrajectoryEncoder()
        self.initial_prior_mean = nn.Parameter(torch.zeros(1))
        self.initial_prior_log_std = nn.Parameter(torch.tensor([math.log(initial_prior_std)]))
        self.initial_posterior = nn.Linear(self.encoder.context_size, 2)
        # Match the initial prior before the encoder learns from data.
        nn.init.zeros_(self.initial_posterior.weight)
        nn.init.zeros_(self.initial_posterior.bias)
        with torch.no_grad():
            self.initial_posterior.bias[1] = math.log(initial_prior_std)
        self.drift = DriftNetwork(width)
        self.diffusion = DiffusionNetwork(width)
        self.posterior_drift = PosteriorDrift(self.encoder.context_size, width)
        self.decoder = DecoderNetwork(width)
        self.solver_dt = float(solver_dt)
        self.observation_std = float(observation_std)

    def f(self, time: Tensor, state: Tensor) -> Tensor:
        r"""Drift in \(dz=f\,dt+g\,dW\)."""
        return self.drift(expand_time(time, state), state)

    def g(self, time: Tensor, state: Tensor) -> Tensor:
        """Positive diffusion in the standard Neural SDE notation."""
        return self.diffusion(expand_time(time, state), state)

    def h(self, time: Tensor, state: Tensor, context: Tensor) -> Tensor:
        """Posterior drift; its parameters are independent of the prior drift."""
        return self.posterior_drift(expand_time(time, state), state, context)

    def decode(self, latent: Tensor) -> Tensor:
        """Return the means of the Gaussian observation distributions."""
        return self.decoder(latent)

    @torch.no_grad()
    def sample_observations(
        self, latent: Tensor, *, generator: torch.Generator | None = None
    ) -> Tensor:
        """Draw observations conditionally on latent paths."""
        mean = self.decode(latent)
        noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
        return mean + self.observation_std * noise

    def sample_posterior(
        self,
        observations: Tensor,
        times: Tensor,
        *,
        generator: torch.Generator | None = None,
        diagnostic_paths: int = 0,
    ) -> PosteriorSample:
        """Return posterior paths, separate KL terms, and initial parameters.

        This is a smoothing posterior: its context includes all supplied
        observations, so future forecast targets must not be supplied here.
        """
        validate_time_grid(times)
        if observations.ndim != 3 or observations.shape[1:] != (len(times), 1):
            raise ValueError("observations must have shape (paths, times, 1)")
        times = times.to(observations)
        context = self.encoder(observations, times)
        # The final forward and first backward GRU states both see the full
        # sequence; together they parameterize q_phi(z_0 | x).
        width = self.encoder.context_size // 2
        summary = torch.cat((context[:, -1, :width], context[:, 0, width:]), dim=-1)
        mean, raw_log_std = self.initial_posterior(summary).chunk(2, dim=-1)
        log_std = raw_log_std.clamp(-10.0, 5.0)
        prior_log_std = self.initial_prior_log_std.clamp(-10.0, 5.0)
        initial_kl = 0.5 * (
            torch.exp(2.0 * (log_std - prior_log_std))
            + ((mean - self.initial_prior_mean) / prior_log_std.exp()).square()
            - 1.0
            + 2.0 * (prior_log_std - log_std)
        ).sum(dim=-1)
        noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
        initial_state = mean + log_std.exp() * noise
        integrated = self._integrate(
            len(observations), times, initial_state=initial_state, context=context,
            generator=generator, diagnostic_paths=diagnostic_paths,
        )
        return PosteriorSample(
            integrated.paths, initial_kl, integrated.path_kl, mean, raw_log_std,
            integrated.diagnostics,
        )

    def loss(
        self,
        observations: Tensor,
        times: Tensor,
        *,
        kl_weight: float = 1.0,
        generator: torch.Generator | None = None,
        diagnostic_paths: int = 0,
    ) -> LossOutput:
        """Estimate loss per time point and expose its single-sample components."""
        if not math.isfinite(kl_weight) or kl_weight < 0.0:
            raise ValueError("kl_weight must be finite and non-negative")
        posterior = self.sample_posterior(
            observations, times, generator=generator, diagnostic_paths=diagnostic_paths,
        )
        reconstructed = self.decode(posterior.paths)
        # p_psi(x_i | z_i) = Normal(D_psi(z_i), observation_std).
        normalized_error = (reconstructed - observations) / self.observation_std
        negative_log_likelihood = (
            0.5 * normalized_error.square()
            + math.log(self.observation_std)
            + 0.5 * math.log(2.0 * math.pi)
        ).sum(dim=(1, 2))
        loss = (negative_log_likelihood + kl_weight * (posterior.initial_kl + posterior.path_kl)).mean() / len(times)
        return LossOutput(loss, negative_log_likelihood, reconstructed, posterior)

    def _integrate(
        self,
        number_of_paths: int,
        times: Tensor,
        *,
        initial_state: Tensor | None = None,
        context: Tensor | None = None,
        generator: torch.Generator | None = None,
        diagnostic_paths: int = 0,
    ) -> IntegrationResult:
        """Euler--Maruyama paths and a left-point sum of the KL integral."""
        validate_time_grid(times)
        if number_of_paths < 1:
            raise ValueError("number_of_paths must be positive")
        if isinstance(diagnostic_paths, bool) or not isinstance(diagnostic_paths, int) or diagnostic_paths < 0:
            raise ValueError("diagnostic_paths must be a non-negative integer")
        if initial_state is None:
            noise = torch.randn(
                number_of_paths, 1, dtype=times.dtype, device=times.device, generator=generator
            )
            state = self.initial_prior_mean + self.initial_prior_log_std.clamp(-10.0, 5.0).exp() * noise
        else:
            if initial_state.shape != (number_of_paths, 1):
                raise ValueError("initial_state must have shape (paths, 1)")
            state = initial_state
        path_kl = times.new_zeros(number_of_paths)
        paths = [state]
        diagnostics = SolverDiagnostics(times, min(diagnostic_paths, number_of_paths)) if diagnostic_paths else None

        for index, interval in enumerate(times[1:] - times[:-1]):
            # End every interval exactly at its observation time, including on
            # irregular grids. Each substep is at most solver_dt.
            substeps = max(1, math.ceil(interval.item() / self.solver_dt))
            dt = interval / substeps
            for step in range(substeps):
                time = times[index] + step * dt
                drift, diffusion = self.f(time, state), self.g(time, state)
                posterior_drift = None
                if context is not None:
                    current_context = torch.lerp(context[:, index], context[:, index + 1], step / substeps)
                    posterior_drift = self.h(time, state, current_context)
                    control = (posterior_drift - drift) / diffusion
                    path_kl = path_kl + 0.5 * control.square().sum(dim=-1) * dt
                if diagnostics is not None:
                    fields = diagnostics.fields(drift, diffusion, posterior_drift)
                    diagnostics.add(index, dt, fields, self.diffusion.numerical_floor)
                    if step == 0:
                        diagnostics.snapshot(fields)
                # Keep both drifts differentiable in KL, but evolve with h.
                if posterior_drift is not None:
                    drift = posterior_drift
                noise = torch.randn(
                    state.shape, dtype=state.dtype, device=state.device, generator=generator
                )
                state = state + drift * dt + diffusion * dt.sqrt() * noise
            paths.append(state)

        if diagnostics is not None:
            with torch.no_grad():
                last_h = self.h(times[-1], state, context[:, -1]) if context is not None else None
                diagnostics.snapshot(diagnostics.fields(self.f(times[-1], state), self.g(times[-1], state), last_h))
        return IntegrationResult(torch.stack(paths, dim=1), path_kl, diagnostics.finish() if diagnostics else None)

    @torch.no_grad()
    def simulate(self, number_of_paths: int, times: Tensor, *, generator: torch.Generator) -> Tensor:
        """Generate latent prior paths with the same solver, without observations."""
        return self._integrate(number_of_paths, times, generator=generator).paths
