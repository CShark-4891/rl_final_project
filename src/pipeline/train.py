# train.py
import argparse
import os
import random
import numpy as np
import torch
import yaml
import d3rlpy
import pipeline.dataset_utils as dataset_utils


def set_seed(seed: int):
    """Ensure complete reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    d3rlpy.seed(seed)


def train_model(config_path: str, seed: int, model_path: str, env_name: str):
    # 1. Load configuration file
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[!] Config file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Set the deterministic seed
    set_seed(seed)
    print("===================TRAINING.PY====================================\n")
    print(f"Torch Is Available: {torch.cuda.is_available()}\n")
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available(
    ) else "CPU\n")
    print("==================================================================\n")
    print(f"Starting Training | Seed: {seed} | Env: {env_name}\n")
    print("==================================================================\n")

    # 3. Load the dataset (Minari for MuJoCo, d3rlpy's built-in loader for demo envs)
    dataset, env = dataset_utils.load_offline_dataset(env_name)
    alpha_override = 0.0
    if "halfcheetah" in env_name:
        alpha_override = 1.0
    else:
        alpha_override = 5.0  # Default choice for the remaining environments (Walker2d, Hopper, Ant, Pendulum demo)
    print("[!] Alpha Override for CQL Conservative Weight: ", alpha_override, "\n")

    # 4. Initialize the CQL Algorithm using Config class
    cql_config = d3rlpy.algos.CQLConfig(
        actor_learning_rate=config["learning_rate_actor"],
        critic_learning_rate=config["learning_rate_critic"],
        batch_size=config["batch_size"],
        gamma=config["gamma"],
        tau=config["tau"],
        conservative_weight=alpha_override  # conservative_weight maps to alpha in CQL
    )

    # Create the learnable model instance assigned to the requested device
    device = config["device"]
    if device != "cpu" and not torch.cuda.is_available():
        print(
            f"[!] Requested device '{device}' but CUDA is unavailable. Falling back to CPU.\n")
        device = "cpu"
    algo = cql_config.create(device=device)

    # 5. Execute Offline Training Loop
    algo.fit(
        dataset,
        n_steps=config["n_steps"],
        # Creates clean structural checkpoints/logs every 10k steps
        n_steps_per_epoch=10000,
        experiment_name=f"CQL_{env_name}_seed_{seed}",
        with_timestamp=False,
        show_progress=True
    )

    # 6. Save out the model bundle (saves architecture + weights for easy reloading)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    algo.save(model_path)
    print(
        f"[+] Successfully saved trained model architecture and parameters to {model_path}\n")
    print("======================TRAINING.PY END============================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CQL on d3rlpy dataset with a specific seed")
    parser.add_argument(
        "--config_path", type=str, default="configs/cql_default.yaml", help="Path to the config file")
    parser.add_argument("--seed", type=int, required=True,
                        help="Random seed for this training run")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Destination path to save the trained model .d3 file")
    parser.add_argument("--env_name", type=str, required=True,
                        help="Environment name for dataset loading")
    args = parser.parse_args()
    train_model(args.config_path, args.seed, args.model_path, args.env_name)
