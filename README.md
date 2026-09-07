# Latent Neural SDE

This repository contains a one-dimensional latent Neural SDE from clean stochastic-process trajectories.

## Quick start

Install the environment, generate a clean dataset, train the model, and render decoded prior rollouts against the empirical data envelope:

```bash
just sync
just all
```

The default workflow writes the dataset to `artifacts/datasets/default.pt` and the checkpoint, metrics, generated trajectories, CSV, and SVG plot to `artifacts/latent_sde`.

## Training

Train the model from a prepared trajectory dataset:

```bash
uv run neuralsde \
  --config configs/default.train.yaml \
  --dataset artifacts/datasets/default.pt \
  --device cpu \
  --output artifacts/my_experiment
```

The training configuration contains model, optimization, validation, and logging settings. Clean trajectories and their shared time grid are loaded directly from the dataset.

TensorBoard logging is enabled by default. Each run creates its own directory under
`<output>/tensorboard/`. For the default output directory, start the dashboard with:

```bash
uv run tensorboard --logdir artifacts/latent_sde/tensorboard
```

## Synthetic datasets

The sampler groups in [`configs/default.dataset.yaml`](configs/default.dataset.yaml) define the process type, number of paths, and process parameters:

```yaml
samplers:
  - type: ou
    paths: 512
    params:
      theta: 1.2
      mu: -1.0
      sigma: 0.45
```

The reusable `neuralsde.synthetic_data` package provides exact-transition samplers for stochastic processes and returns tensors with shape `(paths, times, 1)`. The default configs use $x_0=0$; the variational initial state can also model datasets with other starting values.

### Generate a dataset

Dataset generation is a separate preparation step. Use the dataset config to generate trajectories once and save them as a portable PyTorch file:

```bash
uv run ./scripts/syntetic.py \
  --config configs/default.dataset.yaml \
  --output artifacts/datasets/default.pt
```

The resulting `.pt` file contains clean `trajectories` with shape `(paths, times, 1)`, the exact `shared_time_grid`, and `generation_config`. Datasets from the earlier noisy `observations` format must be regenerated.

