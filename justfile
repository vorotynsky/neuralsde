dataset_config := "configs/default.dataset.yaml"
train_config := "configs/default.train.yaml"
dataset_path := "artifacts/datasets/default.pt"
train_output := "artifacts/latent_sde"
plot_output := train_output / "sampled_trajectories.svg"

# Show the available commands.
default:
    @just --list

# Install or update the project environment from pyproject.toml.
sync:
    uv sync

# Generate the syntetic dataset with just-in-editor config with random values
generate-interactive:
    uv run scripts/syntetic.py --output artifacts/data.pt

# Generate the synthetic training dataset.
generate dataset_config=dataset_config output=dataset_path:
    uv run scripts/syntetic.py --config "{{ dataset_config }}" --output "{{ output }}"

# Train the latent Neural SDE.
train train_config=train_config dataset=dataset_path device="cpu" output=train_output:
    uv run neuralsde \
        --config "{{ train_config }}" \
        --dataset "{{ dataset }}" \
        --device "{{ device }}" \
        --output "{{ output }}"

# Plot trajectories produced by the training run.
plot input=(train_output / "sampled_trajectories.csv") output=plot_output:
    uv run scripts/plot_trajectories.py "{{ input }}" "{{ output }}"

# Generate data, train the model, and render the trajectories.
all: generate train plot
