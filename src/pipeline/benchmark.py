import argparse
import os
import json
import numpy as np
import d3rlpy
import pipeline.dataset_utils as dataset_utils


def __compute_adjusted_d4rl_score(env_name: str, raw_score: float) -> float:
    """Calculates normalized benchmarking metric standard: 0=Random, 100=Expert."""

    ref = dataset_utils.get_ref_score_from_dataset_name(env_name)
    normalized = 100.0 * \
        (raw_score - ref["random"]) / (ref["expert"] - ref["random"])
    return normalized


def perform_benchmark(config_path: str, model_path: str, seed: int, eval_episodes: int, metrics_path: str, env_name: str):

    print("====================BENCHMARK.PY================================\n")
    # 1. Load context configuration properties TODO: What is this used for?
    # with open(config_path, "r") as f:
    #     config = yaml.safe_load(f)

    print(f"Loading policy model payload from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Target model file missing at {model_path}")

    # Reconstruct graph parameters and device alignments cleanly
    algo = d3rlpy.load_learnable(model_path)

    # 2. Build tracking environment
    # Load the same environment used during training
    dataset, env = dataset_utils.load_offline_dataset(env_name)

    gym_id = env.spec.id if env.spec is not None else env_name

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    env.action_space.seed(seed)

    episode_returns = []

    print(
        f"Starting {eval_episodes} evaluation episodes for {gym_id} (Seed: {seed})...")

    # 3. Simulation evaluation loop
    for ep in range(eval_episodes):
        # Varied but deterministic sub-seeds
        obs, info = env.reset(seed=seed + ep)
        done = False
        truncated = False
        total_reward = 0.0

        while not (done or truncated):
            # d3rlpy models demand a explicit leading batch axis: [1, obs_dim]
            action = algo.predict(np.expand_dims(obs, axis=0))[0]
            obs, reward, done, truncated, info = env.step(action)
            total_reward += float(reward)

        episode_returns.append(total_reward)
        print(
            f"Episode {ep + 1}/{eval_episodes} Finished | Raw Return: {total_reward:.2f}")

    # 4. Calculate final metrics profiles
    mean_raw_score = float(np.mean(episode_returns))
    std_raw_score = float(np.std(episode_returns))
    if dataset_utils.has_ref_score(env_name):
        normalized_d4rl_score = __compute_adjusted_d4rl_score(
            env_name, mean_raw_score)
    else:
        normalized_d4rl_score = None
        print(
            f"[!] No D4RL reference score available for '{env_name}'; skipping normalized-score computation.")
    d4rl_score_str = f"{normalized_d4rl_score:.2f}%" if normalized_d4rl_score is not None else "N/A"
    print("===============================================================\n")
    print(f"FINAL EVALUATION PROFILE FOR SEED {seed} ")
    print("===============================================================\n")
    print(f" Target Environment:       {gym_id}")
    print(
        f" Raw Mean Return:          {mean_raw_score:.2f} ± {std_raw_score:.2f}")
    print(f"D4RL Normalized Score:    {d4rl_score_str}")
    print("===============================================================\n")

    # 5. Export structured telemetry metrics
    results_payload = {
        "env_name": env_name,
        "gym_id": gym_id,
        "evaluation_seed": seed,
        "mean_raw_score": mean_raw_score,
        "std_raw_score": std_raw_score,
        "d4rl_normalized_score": normalized_d4rl_score,
        "all_episode_returns": episode_returns
    }

    if metrics_path is None:
        # Default to saving in the same directory as the model artifact
        model_dir = os.path.dirname(model_path)
        metrics_filename = "metrics_2.json"
        metrics_path = os.path.join(model_dir, metrics_filename)

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(results_payload, f, indent=4)
    print(f"Logged evaluations data matrix cleanly to {metrics_path}")
    print("====================BENCHMARK.PY END============================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a saved d3rlpy policy inside Gymnasium")
    parser.add_argument("--config_path", type=str, default="../configs/cql_default.yaml",
                        help="Path to config file")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to saved .d3 model artifact")
    parser.add_argument("--seed", type=int, required=True,
                        help="Seed for environment evaluation determinism")
    parser.add_argument("--eval_episodes", type=int, default=10,
                        help="Number of simulation episodes to run")
    parser.add_argument("--metrics_path", type=str, default=None,
                        help="Output destination JSON path for tracking data")
    parser.add_argument("--env_name", type=str, required=True,
                        help="Environment name for dataset loading")
    args = parser.parse_args()
    perform_benchmark(args.config_path, args.model_path, args.seed,
                      args.eval_episodes, args.metrics_path, args.env_name)
