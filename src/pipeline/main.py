# main.py
import argparse
import os
import subprocess
import sys
import yaml
import json
import numpy as np
import pipeline.dataset_utils as dataset_utils


def check_agent_benchmark_completion(metrics_path: str) -> bool:
    """Checks if the metrics.json file already exists to avoid redundant benchmarking."""
    result_file = os.path.exists(metrics_path) and os.path.isfile(metrics_path)
    return result_file


def check_agent_training_completion(model_path: str, steps: int) -> bool:
    """Checks if the model artifact already exists to avoid redundant training."""
    result_file = os.path.exists(model_path) and os.path.isfile(model_path)
    return result_file


def run_pipeline(config_path: str, base_output_dir: str, eval_episodes: int, dataset_number: int = 0):
    """Orchestrates multi-seeded decoupled training and benchmarking runs sequentially."""
    # 1. Load primary configurations matrix
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[!] CQL config not found at: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seeds = config.get("seeds", [42, 1337, 2026])
    env_name = dataset_utils.DATASETS[dataset_number]
    converted_dataset_name = dataset_utils.get_path_from_dataset_name(env_name)

    print("-------------------------------------------------------------------------\n")
    # The absolute value formatting handles the LaTeX wrapper replacement rule cleanly
    print(
        f"[+] INITIALIZING OFFLINE EVALUATION PIPELINE FOR {env_name.upper()}")
    print(
        f"[+] Targeting Seeds: {seeds} | Iteration Steps: {config['n_steps']}")
    print("-------------------------------------------------------------------------\n")

    summary_metrics = {}

    # 2. Iterate through seed matrix sequentially
    for seed in seeds:
        print(f"[+] Starting execution pipeline slice for Seed: {seed}\n")

        # Build deterministic directory pathing trees for artifacts
        result_dir = os.path.join(
            base_output_dir, converted_dataset_name, f"seed_{seed}")
        print(f"[+] Output artifacts will be stored in: {result_dir}\n")

        os.makedirs(result_dir, exist_ok=True)
        model_path = os.path.join(result_dir, "model.d3")
        metrics_path = os.path.join(result_dir, "metrics.json")

        # --- Stage A: Isolated Subprocess Training Execution ---
        train_cmd = [
            sys.executable,  # this will be the correct python environment main.py is also launched from
            "train.py",
            "--config_path", config_path,
            "--seed", str(seed),
            "--model_path", model_path,
            "--env_name", env_name
        ]

        print("[1/2] Invoking training worker script...")
        # Check if the model already exists to avoid redundant training
        result = check_agent_training_completion(model_path, config["n_steps"])
        if result:
            print(
                f"[!] Warning: Model artifact already exists for dataset index {dataset_number}. Skipping training and evaluation.")
        else:
            train_process = subprocess.run(train_cmd, check=True)
            if train_process.returncode != 0:
                raise RuntimeError(
                    f"[!] Training failed violently on seed {seed} execution path.")

        # --- Stage B: Isolated Subprocess Benchmarking Evaluation ---
        bench_cmd = [
            sys.executable,
            "benchmark.py",
            "--config_path", config_path,
            "--model_path", model_path,
            "--seed", str(seed),
            "--eval_episodes", str(eval_episodes),
            "--metrics_path", metrics_path,
            "--env_name", env_name
        ]
        print("[2/2] Invoking deterministic simulator benchmarking engine...")
        # Check if the metrics file already exists to avoid redundant benchmarking
        result = check_agent_benchmark_completion(metrics_path)
        if result:
            print(
                f"Warning: Metrics file already exists for dataset index {dataset_number}. Skipping benchmarking evaluation.")
        else:
            bench_process = subprocess.run(bench_cmd, check=True)
            if bench_process.returncode != 0:
                raise RuntimeError(
                    f"Benchmarking suite crashed on seed {seed} profile processing.")

        # --- Stage C: Telemetry Slice Reading ---
        with open(metrics_path, "r") as f:
            seed_data = json.load(f)
            summary_metrics[seed] = seed_data

    # 3. Compile aggregated mathematical analytics across all seeds
    has_ref = dataset_utils.has_ref_score(env_name)
    if has_ref:
        normalized_scores = [meta["d4rl_normalized_score"]
                             for meta in summary_metrics.values()]
        mean_aggregate = float(np.mean(normalized_scores))
        std_aggregate = float(np.std(normalized_scores))
    else:
        mean_aggregate = None
        std_aggregate = None

    # 4. Format a comprehensive summary profile on screen
    print("----------------------------------------------------------------\n")
    print("[!] PIPELINE RUN COMPLETION: AGGREGATED TELEMETRY ")
    print("----------------------------------------------------------------\n")
    for seed, data in summary_metrics.items():
        score_str = f"{data['d4rl_normalized_score']:6.2f}%" if data['d4rl_normalized_score'] is not None else "   N/A"
        print(
            f" Seed {seed:4d} | Raw Mean: {data['mean_raw_score']:8.2f} | D4RL Score: {score_str}")
    print("----------------------------------------------------------------\n")
    print("OVERALL SYSTEM PERFORMANCE PROFILE:")
    if mean_aggregate is not None:
        print(
            f" Aggregate Normalized D4RL Score: {mean_aggregate:.2f}% ± {std_aggregate:.2f}%")
    else:
        print(" Aggregate Normalized D4RL Score: N/A (no D4RL reference for this dataset)")
    print("----------------------------------------------------------------\n")

    # 5. Export comprehensive pipeline data payload
    final_report = {
        "environment": env_name,
        "configuration": config,
        "aggregate_performance": {
            "mean_d4rl_score": mean_aggregate,
            "std_d4rl_score": std_aggregate
        },
        "seed_level_telemetry": summary_metrics
    }

    report_destination = os.path.join(
        base_output_dir, converted_dataset_name, "global_pipeline_report.json")
    with open(report_destination, "w") as f:
        json.dump(final_report, f, indent=4)

    print(
        f"[+] Saved holistic pipeline diagnostics matrix securely to: {report_destination}\n")
    print(
        f"[+] Pipeline execution for dataset index {dataset_number} completed successfully.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Master orchestrator pipeline for multi-seeded offline RL runs")
    parser.add_argument(
        "--config", type=str, default="../configs/cql_default.yaml", help="Path to configuration yaml")
    parser.add_argument("--output_dir", type=str, default="../results/cql_runs",
                        help="Root directory tree destination")
    parser.add_argument("--eval_episodes", type=int, default=10,
                        help="Evaluation episode quantity per seed sweep")
    args = parser.parse_args()

    for dataset_number in range(len(dataset_utils.DATASETS)):

        print("---------------------------------STARTING PIPELINE EXECUTION----------------------------------------\n")
        print(
            f"\n\n[+] Starting Pipeline Execution for Dataset Index: {dataset_utils.DATASETS[dataset_number]}\n")
        run_pipeline(
            config_path=args.config,
            base_output_dir=args.output_dir,
            eval_episodes=args.eval_episodes,
            dataset_number=dataset_number
        )
        print("---------------------------------PIPELINE EXECUTION COMPLETED----------------------------------------\n")
