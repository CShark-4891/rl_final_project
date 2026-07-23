# benchmark.py
import argparse
import os
import json
import numpy as np
import gymnasium as gym
import yaml
import d3rlpy

# Official D4RL reference baselines for normalized scoring calculations
# TODO: Needs Checking!
D4RL_REF_SCORES = {
    "halfcheetah-medium-v2": {"random": -280.05, "expert": 12135.0},
    "halfcheetah-random-v2": {"random": -280.05, "expert": 12135.0},
    "halfcheetah-expert-v2": {"random": -280.05, "expert": 12135.0},
    "halfcheetah-medium-replay-v2": {"random": -280.05, "expert": 12135.0},
    "halfcheetah-medium-expert-v2": {"random": -280.05, "expert": 12135.0},
    "hopper-medium-v2": {"random": -20.0, "expert": 3234.3},
    "hopper-random-v2": {"random": -20.0, "expert": 3234.3},
    "hopper-expert-v2": {"random": -20.0, "expert": 3234.3},
    "walker2d-medium-v2": {"random": 1.62, "expert": 4592.3},
    "walker2d-random-v2": {"random": 1.62, "expert": 4592.3},
    "walker2d-expert-v2": {"random": 1.62, "expert": 4592.3}
}

def get_gym_id(env_name: str) -> str:
    """Mapping to convert internal D4RL strings into Gymnasium keys."""

    # 1. Extract the core environment family from the provided string
    if "hopper" in env_name:
        return "Hopper-v4"
    elif "walker2d" in env_name:
        return "Walker2d-v4"
    elif "halfcheetah" in env_name:
        return "HalfCheetah-v4"
    else:
        raise ValueError(f"[!] Environment family '{env_name}' is unsupported.'")

def compute_d4rl_score(env_name: str, raw_score: float) -> float:
    """Calculates normalized benchmarking metric standard: 0=Random, 100=Expert."""
    result_str = ""

    if "halfcheetah" in env_name:
        result_str = "halfcheetah"
    elif "hopper" in env_name:
        result_str = "hopper"
    elif "walker2d" in env_name:
        result_str = "walker2d"

    if "medium" in env_name:
        result_str += "-medium-v2"
    elif "expert" in env_name:
        result_str += "-expert-v2"
    elif "simple" in env_name:
        result_str += "-simple-v2"

    if result_str not in D4RL_REF_SCORES:
        print(f"[+] Warning: Reference scores for {env_name} missing. Returning raw score.")
        return raw_score

    ref = D4RL_REF_SCORES[result_str]
    normalized = 100.0 * (raw_score - ref["random"]) / (ref["expert"] - ref["random"])
    return normalized

def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved d3rlpy policy inside Gymnasium")
    parser.add_argument("--config", type=str, default="../configs/cql_default.yaml", help="Path to config file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved .d3 model artifact")
    parser.add_argument("--seed", type=int, required=True, help="Seed for environment evaluation determinism")
    parser.add_argument("--eval_episodes", type=int, default=10, help="Number of simulation episodes to run")
    parser.add_argument("--metrics_output", type=str, required=True, help="Output destination JSON path for tracking data")
    parser.add_argument("--environment", type=str, required=True, help="Environment name for dataset loading")
    args = parser.parse_args()

    print(f"====================BENCHMARK.PY====================================\n")
    # 1. Load context configuration properties
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    env_name = args.environment
    gym_id = get_gym_id(env_name)

    print(f" [+] Loading policy model payload from: {args.model_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Target model file missing at {args.model_path}")

    # Reconstruct graph parameters and device alignments cleanly
    algo = d3rlpy.load_learnable(args.model_path)

    # 2. Build tracking environment
    env = gym.make(gym_id)
    env.action_space.seed(args.seed)

    episode_returns = []

    print(f" [+] Starting {args.eval_episodes} evaluation episodes for {gym_id} (Seed: {args.seed})...")

    # 3. Simulation evaluation loop
    for ep in range(args.eval_episodes):
        obs, info = env.reset(seed=args.seed + ep) # Varied but deterministic sub-seeds
        done = False
        truncated = False
        total_reward = 0.0

        while not (done or truncated):
            # d3rlpy models demand a explicit leading batch axis: [1, obs_dim]
            action = algo.predict(np.expand_dims(obs, axis=0))[0]
            obs, reward, done, truncated, info = env.step(action)
            total_reward += float(reward)

        episode_returns.append(total_reward)
        print(f" [+] Episode {ep+1}/{args.eval_episodes} Finished | Raw Return: {total_reward:.2f}")

    # 4. Calculate final metrics profiles
    mean_raw_score = float(np.mean(episode_returns))
    std_raw_score = float(np.std(episode_returns))
    normalized_d4rl_score = compute_d4rl_score(env_name, mean_raw_score)

    print(f"================================================================\n")
    print(f" [+] FINAL EVALUATION PROFILE FOR SEED {args.seed} ")
    print(f"================================================================\n")
    print(f" Target Environment:       {gym_id}")
    print(f" Raw Mean Return:          {mean_raw_score:.2f} ± {std_raw_score:.2f}")
    print(f" D4RL Normalized Score:    {normalized_d4rl_score:.2f}%")
    print(f"================================================================\n")

    # 5. Export structured telemetry metrics
    results_payload = {
        "env_name": env_name,
        "gym_id": gym_id,
        "evaluation_seed": args.seed,
        "mean_raw_score": mean_raw_score,
        "std_raw_score": std_raw_score,
        "d4rl_normalized_score": normalized_d4rl_score,
        "all_episode_returns": episode_returns
    }

    os.makedirs(os.path.dirname(args.metrics_output), exist_ok=True)
    with open(args.metrics_output, "w") as f:
        json.dump(results_payload, f, indent=4)
    print(f" [+] Logged evaluations data matrix cleanly to {args.metrics_output}")
    print(f"====================BENCHMARK.PY END====================================\n")

if __name__ == "__main__":
    main()