# main.py
import argparse
import os
import subprocess
import yaml
import json
import numpy as np


datasets_global = [


        # --- 1. Hopper Suite ---
        "mujoco/hopper/simple-v0",
        "mujoco/hopper/medium-v0",
        "mujoco/hopper/expert-v0",
        "mujoco/hopper/medium-replay-v0",
        # --- 2. Walker2d Suite ---
        "mujoco/walker2d/simple-v0",
        "mujoco/walker2d/medium-v0",
        "mujoco/walker2d/expert-v0",

        # --- 3. HalfCheetah Suite ---
        "mujoco/halfcheetah/simple-v0",
        "mujoco/halfcheetah/medium-v0",
        "mujoco/halfcheetah/expert-v0",

        # --- 4. Ant Suite ---
 #       "mujoco/ant/simple-v0",
 #       "mujoco/ant/medium-v0",
         "mujoco/ant/expert-v0",

        # --- 5. Humanoid Suite ---
  #      "mujoco/humanoid/simple-v0",
  #      "mujoco/humanoid/medium-v0",
  #      "mujoco/humanoid/expert-v0"
    ]

def check_agent_benchmark_completion(metrics_path: str) -> bool:
    """Checks if the metrics.json file already exists to avoid redundant benchmarking."""
    result_file = os.path.exists(metrics_path) and os.path.isfile(metrics_path)
    return result_file

def check_agent_training_completion(model_path: str, steps: int) -> bool:
    """Checks if the model artifact already exists to avoid redundant training."""
    result_file = os.path.exists(model_path) and os.path.isfile(model_path)
    return result_file

def build_dataset_name(dataset_str: str) -> str:
    result_str = ""
    if "hopper" in dataset_str:
        result_str = "hopper"
    elif "walker2d" in dataset_str:
        result_str = "walker2d"
    elif "halfcheetah" in dataset_str:
        result_str = "halfcheetah"
    elif "ant" in dataset_str:
        result_str = "ant"
    elif "humanoid" in dataset_str:
        result_str = "humanoid"
    elif result_str == "":
        print(f"[!] Warning: Unrecognized dataset string '{dataset_str}'!")
        exit(1)
    if "expert" in dataset_str:
        result_str += "/expert"
    elif "simple" in dataset_str:
        result_str += "/simple"
    elif "medium-replay" in dataset_str:
        result_str += "/medium-replay"
    elif "medium" in dataset_str:
        result_str += "/medium"
    return result_str


def run_pipeline(config_path: str, base_output_dir: str, eval_episodes: int, dataset_number: int = 0):
    """Orchestrates multi-seeded decoupled training and benchmarking runs sequentially."""
    # 1. Load primary configurations matrix
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[!] CQL config not found at: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seeds = config.get("seeds", [42, 1337, 2026])
    env_name = datasets_global[dataset_number]

    print("-------------------------------------------------------------------------\n")
    # The absolute value formatting handles the LaTeX wrapper replacement rule cleanly
    print(f"[+] INITIALIZING OFFLINE EVALUATION PIPELINE FOR {env_name.upper()}")
    print(f"[+] Targeting Seeds: {seeds} | Iteration Steps: {config['n_steps']}")
    print("-------------------------------------------------------------------------\n")

    summary_metrics = {}

    # 2. Iterate through seed matrix sequentially
    for seed in seeds:
        print(f"[+] Starting execution pipeline slice for Seed: {seed}\n")

        # Build deterministic directory pathing trees for artifacts
        converted_dataset_name = build_dataset_name(env_name)
        result_dir = os.path.join(base_output_dir, converted_dataset_name, f"seed_{seed}")
        print(f"[+] Output artifacts will be stored in: {result_dir}\n")

        os.makedirs(result_dir, exist_ok=True)
        model_path = os.path.join(result_dir, "model.d3")
        metrics_path = os.path.join(result_dir, "metrics.json")
        
        # --- Stage A: Isolated Subprocess Training Execution ---
        train_cmd = [
            "python", "train.py",
            "--config", config_path,
            "--seed", str(seed),
            "--output_path", model_path,
            "--environment", datasets_global[dataset_number]
        ]

        print(f"[1/2] Invoking training worker script...")
        # Check if the model already exists to avoid redundant training
        result = check_agent_training_completion(model_path, config["n_steps"])
        if result:
            print(f"[!] Warning: Model artifact already exists for dataset index {dataset_number}. Skipping training and evaluation.")
        else:
            train_process = subprocess.run(train_cmd, check=True)
            if train_process.returncode != 0:
                raise RuntimeError(f"[!] Training failed violently on seed {seed} execution path.")

        # --- Stage B: Isolated Subprocess Benchmarking Evaluation ---
        bench_cmd = [
            "python", "benchmark.py",
            "--config", config_path,
            "--model_path", model_path,
            "--seed", str(seed),
            "--eval_episodes", str(eval_episodes),
            "--metrics_output", metrics_path,
            "--environment", datasets_global[dataset_number]
        ]
        print(f"[2/2] Invoking deterministic simulator benchmarking engine...")
        # Check if the metrics file already exists to avoid redundant benchmarking
        result =check_agent_benchmark_completion(metrics_path)
        if result:
            print(f"[!] Warning: Metrics file already exists for dataset index {dataset_number}. Skipping benchmarking evaluation.")
        else:
            bench_process = subprocess.run(bench_cmd, check=True)
            if bench_process.returncode != 0:
                raise RuntimeError(f"[!] Benchmarking suite crashed on seed {seed} profile processing.")

        # --- Stage C: Telemetry Slice Reading ---
        with open(metrics_path, "r") as f:
            seed_data = json.load(f)
            summary_metrics[seed] = seed_data


    # 3. Compile aggregated mathematical analytics across all seeds
    normalized_scores = [meta["d4rl_normalized_score"] for meta in summary_metrics.values()]
    mean_aggregate = np.mean(normalized_scores)
    std_aggregate = np.std(normalized_scores)

    # 4. Format a comprehensive summary profile on screen
    print("-------------------------------------------------------------------------\n")
    print("[!] PIPELINE RUN COMPLETION: AGGREGATED TELEMETRY ")
    print("-------------------------------------------------------------------------\n")
    for seed, data in summary_metrics.items():
        print(f" Seed {seed:4d} | Raw Mean: {data['mean_raw_score']:8.2f} | D4RL Score: {data['d4rl_normalized_score']:6.2f}%")
    print("-------------------------------------------------------------------------\n")
    print(f" OVERALL SYSTEM PERFORMANCE PROFILE:")
    print(f" Aggregate Normalized D4RL Score: {mean_aggregate:.2f}% ± {std_aggregate:.2f}%")
    print("-------------------------------------------------------------------------\n")

    # 5. Export comprehensive pipeline data payload
    final_report = {
        "environment": env_name,
        "configuration": config,
        "aggregate_performance": {
            "mean_d4rl_score": float(mean_aggregate),
            "std_d4rl_score": float(std_aggregate)
        },
        "seed_level_telemetry": summary_metrics
    }

    report_destination = os.path.join(base_output_dir, converted_dataset_name, "global_pipeline_report.json")
    with open(report_destination, "w") as f:
        json.dump(final_report, f, indent=4)

    print(f"[+] Saved holistic pipeline diagnostics matrix securely to: {report_destination}\n")
    print(f"[+] Pipeline execution for dataset index {dataset_number} completed successfully.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master orchestrator pipeline for multi-seeded offline RL runs")
    parser.add_argument("--config", type=str, default="../configs/cql_default.yaml", help="Path to configuration yaml")
    parser.add_argument("--output_dir", type=str, default="../results/cql_runs", help="Root directory tree destination")
    parser.add_argument("--eval_episodes", type=int, default=10, help="Evaluation episode quantity per seed sweep")
    args = parser.parse_args()

    for dataset_number in range(len(datasets_global)):

        print("---------------------------------STARTING PIPELINE EXECUTION----------------------------------------\n")
        print(f"\n\n[+] Starting Pipeline Execution for Dataset Index: {datasets_global[dataset_number]}\n")
        run_pipeline(
            config_path=args.config,
            base_output_dir=args.output_dir,
            eval_episodes=args.eval_episodes,
            dataset_number=dataset_number
        )
        print("---------------------------------PIPELINE EXECUTION COMPLETED----------------------------------------\n")