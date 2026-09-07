# TensorBoard logging

Install dependencies with `uv sync`, then train normally. Logging is enabled by
default; each run is isolated under `<output>/tensorboard/<UTC timestamp>-<unique id>`.

```bash
uv run tensorboard --logdir artifacts/latent_sde/tensorboard
```

## Configuration and cadence

```yaml
validation_samples: 32
tensorboard:
  enabled: true
  batch_every: 10
  diagnostics_every: 25
  prior_samples: 1024
  posterior_probe_paths: 64
  plot_paths: 8
```

All counts and intervals must be positive integers. `enabled` must be a boolean.
Unknown nested keys are rejected. `validation_samples` is the number of independent
posterior paths **per validation example**, not per batch or dataset. It can be raised,
for example, to 40. Validation cost scales approximately with this count.

| Cadence | Records | Horizontal axis |
| --- | --- | --- |
| Start | Configuration, device, model, parameter count, package versions, dataset generation metadata, split indices, exact time grid, RNG conventions | Step 0 |
| First optimizer step and every `batch_every` steps | Current batch metrics and gradient norms | Global optimizer step |
| Every epoch | Train/validation metrics, initial distributions, parameter norms, optimization summaries, timings | Epoch |
| First, last, and every `diagnostics_every` epoch | Prior generation metrics, solver diagnostics, histograms, figures | Epoch |
| End or failure | Status, last epoch/step, error if any | Global optimizer step |

Batch records describe the current batch, not a moving average. `log_every` only controls
console output. Disabling TensorBoard also disables extra prior/solver probes and figures,
but keeps the same Monte Carlo validation every epoch.

Metadata and final status are also stored in `metadata.json` alongside event files.
`metrics.json` retains `training.train_loss` and `training.val_loss`, and adds full
`train_metrics`, `validation_metrics`, `generation_metrics`, and `tensorboard_dir`.
Generation metrics are empty when TensorBoard diagnostics are disabled.

## Objective and reconstruction

For a trajectory with $M$ observations, let $R$ be its summed Gaussian negative
log-likelihood, $K_0$ the analytic initial KL, $K_p$ the sampled path KL, and $\beta$
the configured `kl_weight`. The differentiable training objective per time point is

$$\ell = \frac{R + \beta(K_0+K_p)}{M}.$$

The following tags have prefixes `train/step`, `train/epoch`, and `validation/epoch`.
Epoch averages weight examples equally, including the final partial batch. Validation
also averages the posterior draws equally. Time-point averages weight grid points equally,
even for irregular grids; this is separate from the time integral used for path KL.

| Suffix | Definition |
| --- | --- |
| `loss` | Optimized objective $\ell$ |
| `negative_elbo` | $(R+K_0+K_p)/M$, independent of the chosen KL weight |
| `nll` | $R/M$ |
| `kl_initial`, `kl_path` | $K_0/M$, $K_p/M$ |
| `kl_total`, `kl_weighted` | $(K_0+K_p)/M$, $\beta(K_0+K_p)/M$ |
| `kl_initial_per_path`, `kl_path_per_path` | Unnormalized $K_0$, $K_p$ |
| `reconstruction_mse` | Mean squared decoded-sample error over examples, times, and samples |
| `reconstruction_rmse` | Square root of aggregate MSE, not an average of batch RMSEs |
| `reconstruction_mae` | Mean absolute decoded-sample error |

The invariant is `loss = nll + kl_weighted`. A continuous Gaussian density can yield
negative NLL, so a negative loss is not itself a failure. Train metrics use the successive
model states during optimization; validation uses the model after the epoch.

Additional `validation/epoch` tags:

| Suffix | Definition |
| --- | --- |
| `posterior_mean_mse`, `posterior_mean_rmse` | Error of the decoded mean across posterior draws |
| `posterior_std_mean` | Average population standard deviation across decoded draws |
| `loss_mc_se`, `nll_mc_se`, `kl_path_mc_se` | Monte Carlo standard errors of the complete validation-set means |

For $S$ draws, compute one whole-dataset mean $\bar\ell_s$ per draw index, then

$$\mathrm{SE}_{\mathrm{MC}} =
\sqrt{\frac{\sum_{s=1}^S(\bar\ell_s-\bar\ell)^2}{S(S-1)}}.$$

This quantifies sampling error for the fixed validation set, not uncertainty from selecting
that dataset. Standard-error tags are omitted for $S=1$; the empirical posterior spread
is then zero and does not establish posterior certainty. Loss always averages sampled
likelihoods, not the likelihood of the mean reconstruction. These are smoothing
reconstructions: the encoder sees the entire observed sequence, including its future.

## Optimization and runtime

- `optimization/step`: `learning_rate`, `kl_weight`, `grad_norm_before_clip`,
  `grad_norm_after_clip`. Clipping uses the existing global norm threshold of 5.
- `optimization/step/grad_norm/<module>`: pre-clipping norms for `encoder`,
  `initial_posterior`, `drift`, `posterior_drift`, `diffusion`, `decoder`, `initial_prior`.
- `optimization/epoch`: `grad_norm_before_clip_mean`, `grad_norm_before_clip_max`,
  `grad_norm_after_clip_mean`, `clipped_step_fraction`. These aggregate **all optimizer
  steps**, equally weighted, rather than only the logged steps.
- `parameters/epoch/norm/<module>`: parameter norms after the epoch.
- `performance`: `train_seconds`, `validation_seconds`, `diagnostics_seconds`,
  `epoch_seconds`, `train_paths_per_second`, `validation_samples_per_second`.
  Validation throughput counts $N_{\mathrm{val}}S$ integrations. Training time includes
  batch logging; diagnostics time includes epoch logging and figures. CUDA is synchronized
  at phase boundaries. Final performance-event writes are outside the measured epoch.

## Prior generation quality

`generation/*` compares independent decoded prior paths with all validation paths at
the same observation times. No observation noise is added: the current dataset contains
clean trajectories, and these diagnostics evaluate decoded means.

| Suffix | Definition |
| --- | --- |
| `mean_rmse` | RMSE of the two time-dependent ensemble means |
| `std_rmse` | RMSE of population standard deviations across paths at each time |
| `marginal_w1_mean`, `marginal_w1_max` | Mean and maximum empirical Wasserstein-1 distance across observation times |
| `terminal_w1` | Wasserstein-1 distance at the final time |
| `increment_w1_mean` | Mean Wasserstein-1 distance between adjacent-increment distributions on matching intervals |
| `covariance_rmse` | Entrywise RMSE of temporal covariance matrices, using population normalization |
| `quadratic_variation_mean_error` | Absolute difference between means of $\sum_i(x_{i+1}-x_i)^2$ |

Wasserstein-1 is computed exactly for the two empirical distributions, including unequal
sample sizes, as the integral of the absolute difference of their CDFs. Time averages
are arithmetic. Increment comparisons preserve the actual observation intervals;
covariance does not assume stationarity. The quadratic-variation diagnostic is a
finite-grid statistic, not a claim about a continuous-time limit.

There is no paired MSE between independent generated and observed paths.
When either ensemble has fewer than two paths, std/covariance comparisons and the
covariance figure are omitted, with the reason recorded in metadata.

## SDE and decoder diagnostics

`dynamics/posterior` uses the first `posterior_probe_paths` validation examples and all
their existing validation samples; it does not draw another set of posterior paths.
`dynamics/prior` uses the diagnostic prior paths. Define

$$u = (h-f)/g, \qquad \tau=t_{M-1}-t_0.$$

Means and RMS values below are weighted by actual solver-substep duration, then averaged
over paths. Min/max values range over all left endpoints used by the solver.

| Prefix | Suffixes |
| --- | --- |
| `dynamics/posterior` | `prior_drift_rms`, `posterior_drift_rms`, `drift_gap_rms`, `control_rms` |
| `dynamics/prior` | `prior_drift_rms` |
| Both dynamics prefixes | `diffusion_mean`, `diffusion_min`, `diffusion_max`, `diffusion_near_floor_fraction` |
| `initial/prior` | `mean`, raw `log_std`, effective `std` after clamping |
| `initial/posterior` | `mean_mean`, `mean_std`, `std_mean`, `std_min`, `std_max`, `lower_clamp_fraction`, `upper_clamp_fraction` |
| `decoder` | `posterior_residual_rms`, `prior_residual_rms` of $D(z)-z$ on observation times |

The near-floor fraction is the path/time fraction with $g<10\,g_{\mathrm{floor}}$.
Initial-posterior statistics use every validation example once, without Monte Carlo
duplication. Clamp fractions count raw log standard deviations below -10 or above 5.

The time-weighted control satisfies

$$\mathbb{E}[K_p]=\frac{\tau}{2}\operatorname{RMS}(u)^2$$

on the same probed trajectories. Drift and diffusion are latent-coordinate diagnostics;
they are not directly compared with the generating processes' coefficients.

## Figures and histograms

| Figure tag under `figures/` | Contents |
| --- | --- |
| `posterior_reconstruction` | Fixed validation examples, decoded posterior mean, 5–95% quantiles, first three sampled paths |
| `reconstruction_error` | Sample MSE and decoded posterior standard deviation over time, using all validation examples |
| `prior_paths` | Independent decoded prior paths |
| `prior_marginals` | Ensemble means and 5–95% bands for prior and validation |
| `marginal_distributions` | Empirical CDFs at initial, middle-grid, and final times; duplicate indices are removed |
| `temporal_covariance` | Validation/prior covariance and their difference on the actual time grid |
| `dynamics` | Interval-wise time-weighted drift RMS and diffusion means |
| `path_kl_density` | Mean interval path KL divided by interval duration |

Posterior bands describe the smoothing distribution of decoded means, without
observation noise. They are not forecast coverage estimates.

`histograms/parameters/<name>` records post-epoch parameters;
`histograms/gradients_pre_clip/<name>` records gradients from the final training batch,
before clipping. Other histograms include initial posterior means/stds, prior/posterior
latent and decoded values, reconstruction errors, and drift/diffusion/control values
**on observation times**. These flattened histograms weight grid points equally, unlike
the time-weighted solver scalars. Each histogram retains at most 16,384 evenly spaced
values from its full stream. Scalar reductions always use all values in their probe.

## Reproducibility, memory, and Python interfaces

Training, DataLoaders, validation, prior probes, and final rollouts use separate RNG
streams. Validation and prior diagnostic streams reset to fixed seeds each epoch,
providing common random numbers for comparing model states. Changing TensorBoard
cadence or disabling it does not change training or validation outputs. Reproducibility
is for a fixed device, software environment, batch size, and sample count.

Validation processes one draw and one batch at a time, accumulating moments and
whole-dataset replicate sums. Complete draws are retained only for plotted examples.
Histograms are bounded and diagnostic tensors are detached. Writer data is flushed
each epoch and the writer is closed on success or error. A non-finite loss or gradient
terminates training before the optimizer update, with failure location in run metadata.

The model interfaces now return dataclasses:

- `model.loss(...) -> LossOutput`: normalized scalar `loss`, unnormalized
  `nll_per_path`, `reconstruction`, and `posterior`.
- `model.sample_posterior(...) -> PosteriorSample`: `paths`, `initial_kl`, `path_kl`,
  `initial_mean`, raw `initial_log_std`, derived `initial_std`, optional `diagnostics`.
- `validate(..., samples=32, seed=7, probe_paths=0, plot_paths=0) -> ValidationResult`:
  scalar metrics, initial statistics, optional bounded histogram/figure data, dynamics.

```python
output = model.loss(observations, times, kl_weight=1.0)
output.loss.backward()  # Already normalized per time point; do not divide again.
```

The model contains no TensorBoard writer. Checkpoint parameter names and shapes are
unchanged; checkpoints now include the additional serialized training configuration.
