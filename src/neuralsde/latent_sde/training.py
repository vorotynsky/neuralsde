import csv
import json
import math
from pathlib import Path
from time import perf_counter

import torch
from torch import Tensor
from torch.utils.data import DataLoader, random_split

from .config import ExperimentConfig
from .data import TrajectoryDataset
from .evaluation import validate, trajectories_from_batch, validate_trajectories
from .metrics import MetricMeans, loss_values
from .model import NeuralLatentSDE
from .reporting import TensorBoardLogger, prior_diagnostics


def _clock(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return perf_counter()


def _norm(tensors) -> float:
    tensors = [tensor.detach() for tensor in tensors if tensor is not None]
    return torch.stack([tensor.square().sum() for tensor in tensors]).sum().sqrt().item() if tensors else 0.0


def _groups(model: NeuralLatentSDE) -> dict[str, list[Tensor]]:
    result = {name: list(getattr(model, name).parameters()) for name in
              ("encoder", "initial_posterior", "drift", "posterior_drift", "diffusion", "decoder")}
    result["initial_prior"] = [model.initial_prior_mean, model.initial_prior_log_std]
    return result


def train(
    config: ExperimentConfig,
    dataset: TrajectoryDataset,
    *,
    device: str = "cpu",
    output: str | Path = "artifacts/latent_sde",
) -> dict[str, object]:
    """Train the model, evaluate it, and write its reproducible artifacts."""
    torch.manual_seed(config.seed)
    device = torch.device(device)
    times = dataset.shared_time_grid.to(device)

    if len(dataset) < 2:
        raise ValueError("at least two trajectories are required for training and validation")

    validation_size = (
        config.validation_size
        if isinstance(config.validation_size, int)
        else math.ceil(config.validation_size * len(dataset))
    )

    if validation_size >= len(dataset):
        raise ValueError("validation_size must leave at least one training trajectory")

    training_dataset, validation_dataset = random_split(
        dataset,
        [len(dataset) - validation_size, validation_size],
        generator=torch.Generator().manual_seed(config.seed + 4),
    )

    loader = DataLoader(
        training_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed + 2),
    )

    validation_loader = DataLoader(
        validation_dataset, batch_size=config.batch_size,
        generator=torch.Generator().manual_seed(config.seed + 5),
    )
    # Do not advance the shuffled loader or its generator just to save examples.
    example_trajectories = dataset.tensors[0][training_dataset.indices[:config.batch_size]].to(device)
    validate_trajectories(example_trajectories, len(times))

    model = NeuralLatentSDE(
        solver_dt=config.solver_dt,
        observation_std=config.observation_std,
        initial_prior_std=config.initial_prior_std,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    tb = config.tensorboard
    notes = []
    if min(len(validation_dataset), tb.prior_samples) < 2:
        notes.append("Generation std/covariance metrics and covariance figure require at least two paths in each set.")
    if config.validation_samples < 2:
        notes.append("Monte Carlo standard errors require at least two posterior samples.")
    logger = TensorBoardLogger(output, config, {
        "device": str(device), "model": repr(model),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_paths": len(training_dataset), "validation_paths": len(validation_dataset),
        "train_indices": training_dataset.indices, "validation_indices": validation_dataset.indices,
        "time_grid": times.cpu().tolist(), "generation_config": dataset.generation_config,
        "notes": notes,
        "rng_offsets": {"split": 4, "train_loader": 2, "validation_loader": 5,
                        "validation_sample": "1000 + sample_index", "prior_diagnostics": 2000, "final_rollout": 3},
    })
    epoch, global_step = 0, 0
    error = None
    generation = {}
    try:
        for epoch in range(1, config.epochs + 1):
            started = _clock(device)
            diagnostic_epoch = tb.enabled and (epoch == 1 or epoch == config.epochs or epoch % tb.diagnostics_every == 0)
            model.train()
            means = MetricMeans()
            before_sum, after_sum, before_max, clipped, steps = 0.0, 0.0, 0.0, 0, 0
            gradient_histograms = {}
            for batch_index, batch in enumerate(loader):
                global_step += 1
                steps += 1
                trajectories = trajectories_from_batch(batch).to(device)
                validate_trajectories(trajectories, len(times))
                optimizer.zero_grad(set_to_none=True)
                loss_output = model.loss(trajectories, times, kl_weight=config.kl_weight)
                if not torch.isfinite(loss_output.loss):
                    raise FloatingPointError("non-finite training loss")
                values = loss_values(loss_output, trajectories, config.kl_weight)
                means.add(values)
                loss_output.loss.backward()
                recorded_step = tb.enabled and (global_step == 1 or global_step % tb.batch_every == 0)
                if recorded_step:
                    module_grad_norms = {
                        name: _norm(parameter.grad for parameter in parameters) for name, parameters in _groups(model).items()
                    }
                if diagnostic_epoch and batch_index == len(loader) - 1:
                    gradient_histograms = {f"gradients_pre_clip/{name}": parameter.grad.detach().cpu().clone()
                                           for name, parameter in model.named_parameters() if parameter.grad is not None}
                before = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True).item()
                after = _norm(parameter.grad for parameter in model.parameters())
                optimizer.step()
                before_sum += before
                after_sum += after
                before_max = max(before_max, before)
                clipped += int(before > 5.0)
                if recorded_step:
                    logger.scalars("optimization/step/grad_norm", module_grad_norms, global_step)
                    batch_metrics = MetricMeans()
                    batch_metrics.add(values)
                    logger.scalars("train/step", batch_metrics.result(), global_step)
                    logger.scalars("optimization/step", {
                        "learning_rate": optimizer.param_groups[0]["lr"], "kl_weight": config.kl_weight,
                        "grad_norm_before_clip": before, "grad_norm_after_clip": after,
                    }, global_step)
                # No autograd-backed outputs survive into the next forward pass.
                del loss_output, values
            train_end = _clock(device)
            train_metrics = means.result()
            validation = validate(
                model, validation_loader, times, kl_weight=config.kl_weight, device=device,
                samples=config.validation_samples, seed=config.seed,
                probe_paths=tb.posterior_probe_paths if diagnostic_epoch else 0,
                plot_paths=tb.plot_paths if diagnostic_epoch else 0,
            )
            validation_end = _clock(device)
            logger.scalars("train/epoch", train_metrics, epoch)
            logger.scalars("validation/epoch", validation.metrics, epoch)
            logger.scalars("optimization/epoch", {
                "grad_norm_before_clip_mean": before_sum / steps, "grad_norm_before_clip_max": before_max,
                "grad_norm_after_clip_mean": after_sum / steps, "clipped_step_fraction": clipped / steps,
            }, epoch)
            logger.scalars("parameters/epoch/norm", {name: _norm(parameters) for name, parameters in _groups(model).items()}, epoch)
            logger.scalars("initial/posterior", validation.initial_metrics, epoch)
            logger.scalars("initial/prior", {
                "mean": model.initial_prior_mean.item(), "log_std": model.initial_prior_log_std.item(),
                "std": model.initial_prior_log_std.detach().clamp(-10, 5).exp().item(),
            }, epoch)
            if diagnostic_epoch:
                reference = dataset.tensors[0][validation_dataset.indices]
                prior = prior_diagnostics(model, times, reference, samples=tb.prior_samples,
                                          batch_size=config.batch_size, seed=config.seed)
                generation = prior.metrics
                logger.scalars("generation", generation, epoch)
                logger.scalars("dynamics/posterior", validation.dynamics.metrics(times), epoch)
                logger.scalars("dynamics/prior", prior.dynamics.metrics(times), epoch)
                logger.scalars("decoder", {"posterior_residual_rms": validation.decoder_residual_rms,
                                           "prior_residual_rms": prior.decoder_residual_rms}, epoch)
                logger.histograms({f"parameters/{name}": value for name, value in model.named_parameters()}, epoch)
                logger.histograms(gradient_histograms, epoch)
                logger.histograms({**validation.histograms, **prior.histograms}, epoch)
                logger.figures(times, validation, prior, reference, epoch, tb.plot_paths)
            logger.flush()
            finished = _clock(device)
            logger.scalars("performance", {
                "train_seconds": train_end - started, "validation_seconds": validation_end - train_end,
                "diagnostics_seconds": finished - validation_end, "epoch_seconds": finished - started,
                "train_paths_per_second": len(training_dataset) / (train_end - started),
                "validation_samples_per_second": len(validation_dataset) * config.validation_samples / (validation_end - train_end),
            }, epoch)
            logger.flush()
            if epoch == 1 or epoch % config.log_every == 0 or epoch == config.epochs:
                print(f"Epoch {epoch:4d} | train loss: {train_metrics['loss']:.4f} | val loss: {validation.metrics['loss']:.4f}")

        model.eval()
        rollout_generator = torch.Generator(device=device).manual_seed(config.seed + 3)
        with torch.no_grad():
            generated_latent = model.simulate(config.sample_path, times, generator=rollout_generator)
            generated_decoded = model.decode(generated_latent)
        result: dict[str, object] = {
            "training": {"train_loss": train_metrics["loss"], "val_loss": validation.metrics["loss"],
                         "train_paths": len(training_dataset), "val_paths": len(validation_dataset), "epochs": config.epochs},
            "train_metrics": train_metrics, "validation_metrics": validation.metrics,
            "generation_metrics": generation,
            "tensorboard_dir": str(logger.run_dir) if logger.run_dir else None,
        }
        write_artifacts(output, config, result, model, times, generated_latent, generated_decoded, example_trajectories)
        return result
    except BaseException as exception:
        error = f"{type(exception).__name__}: {exception} (epoch={epoch}, step={global_step})"
        raise
    finally:
        logger.close(status="failed" if error else "completed", epoch=epoch, step=global_step, error=error)


def write_artifacts(
    output: str | Path,
    config: ExperimentConfig,
    result: dict[str, object],
    model: NeuralLatentSDE,
    times: torch.Tensor,
    generated_latent: torch.Tensor,
    generated_decoded: torch.Tensor,
    trajectories: torch.Tensor,
) -> None:
    """Persist the trained model, trajectories, tabular samples, and metrics."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "times": times.cpu(), "config": config.to_dict()},
        output / "latent_neural_sde.pt",
    )
    torch.save(
        {
            "times": times.cpu(),
            "learned_prior_latent_paths": generated_latent.cpu(),
            "learned_prior_decoded_paths": generated_decoded.cpu(),
            "training_trajectories": trajectories[:32].cpu(),
        },
        output / "sampled_trajectories.pt",
    )
    with (output / "sampled_trajectories.csv").open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(("path_id", "time", "latent_value", "decoded_value"))
        for path_id, (latent_path, decoded_path) in enumerate(
            zip(generated_latent.cpu(), generated_decoded.cpu())
        ):
            for time, latent_value, decoded_value in zip(
                times.cpu(), latent_path[:, 0], decoded_path[:, 0]
            ):
                writer.writerow(
                    (path_id, time.item(), latent_value.item(), decoded_value.item())
                )
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
