import os
import json
import glob
import pandas as pd
import numpy as np

# D4RL reference baselines for normalizing raw scores
D4RL_REF_SCORES = {
    "halfcheetah": {"random": -280.05, "expert": 12135.0},
    "hopper":      {"random": -20.0,    "expert": 3234.3},
    "walker2d":    {"random": 1.62,     "expert": 4592.3},
    "ant":         {"random": -325.6,   "expert": 3818.5},
}


def _normalize_raw_score(env_family: str, raw_score: float) -> float | None:
    """Normalize a raw return to a D4RL percentage using reference bounds."""
    if env_family not in D4RL_REF_SCORES:
        return None
    ref = D4RL_REF_SCORES[env_family]
    denom = ref["expert"] - ref["random"]
    if denom == 0:
        return None
    return 100.0 * (raw_score - ref["random"]) / denom


def main():
    print("=========================================================")
    print("[+] Consolidating Evaluation Targets and Diagnostic Metrics")
    print("=========================================================")

    # Path setups relative to src/pipeline/
    results_base_dir = os.path.join("..", "results", "cql_runs")
    profiles_dir = os.path.join("..", "dataset_profiles")
    output_csv = os.path.join("..", "dataset_profiles",
                              "meta_analysis_registry.csv")

    rows = []

    if not os.path.exists(results_base_dir):
        print(
            f"[!] Error: Evaluation results folder not found at {results_base_dir}")
        return
    if not os.path.exists(profiles_dir):
        print(
            f"[!] Error: Diagnostic profiles folder not found at {profiles_dir}")
        return

    print("[+] Gathering evaluation returns from consolidated JSONs...")

    # Loop over environments (e.g., halfcheetah, hopper, walker2d)
    for env_name in os.listdir(results_base_dir):
        env_path = os.path.join(results_base_dir, env_name)
        if not os.path.isdir(env_path):
            continue

        # Loop over dataset tiers (e.g., simple, medium, expert)
        for tier_name in os.listdir(env_path):
            tier_path = os.path.join(env_path, tier_name)
            if not os.path.isdir(tier_path):
                continue

            # Look for the consolidated json file in this directory (e.g., global_pipeline_report.json)
            json_files = glob.glob(os.path.join(tier_path, "*.json"))
            if not json_files:
                print(
                    f"    [!] Warning: No metrics JSON found in {env_name}/{tier_name}")
                continue

            metrics_json = json_files[0]
            normalized_target = None

            try:
                with open(metrics_json, "r") as f:
                    data = json.load(f)

                # Re-normalize from raw scores rather than trusting the stored
                # mean_d4rl_score, which may have been computed with missing
                # reference values (e.g., simple-v2 datasets).
                seed_data = data.get("seed_level_telemetry", {})
                if seed_data:
                    # Determine env family from the environment string
                    env_name_str = data.get("environment", "")
                    env_family = None
                    for ef in D4RL_REF_SCORES:
                        if ef in env_name_str.lower():
                            env_family = ef
                            break

                    if env_family:
                        raw_scores = [
                            t["mean_raw_score"]
                            for t in seed_data.values()
                            if t.get("mean_raw_score") is not None
                        ]
                        if raw_scores:
                            mean_raw = float(np.mean(raw_scores))
                            normalized_target = _normalize_raw_score(env_family, mean_raw)

                # Fallback: use stored mean_d4rl_score if re-normalization failed
                if normalized_target is None:
                    agg = data.get("aggregate_performance", {})
                    if "mean_d4rl_score" in agg:
                        normalized_target = agg["mean_d4rl_score"]
                    else:
                        scores = [
                            telemetry["d4rl_normalized_score"]
                            for telemetry in seed_data.values()
                        ]
                        if scores:
                            normalized_target = float(np.mean(scores))

            except Exception as e:
                print(
                    f"    [!] Failed to parse benchmark metrics for {env_name}/{tier_name}: {e}")
                continue

            if normalized_target is None:
                continue

            # --- PAIR WITH STRUCTURAL DIAGNOSTIC PROFILE FEATURES ---
            # Standardize naming alignment (e.g., mujoco_halfcheetah_expert-v0.json)
            profile_name = f"mujoco_{env_name}_{tier_name}-v0.json"
            profile_file = os.path.join(profiles_dir, profile_name)

            # Fallback dynamic match if versions or names change slightly
            if not os.path.exists(profile_file):
                profile_pattern = os.path.join(
                    profiles_dir, f"*{env_name}*{tier_name}*.json")
                matching_profiles = glob.glob(profile_pattern)
                if matching_profiles:
                    profile_file = matching_profiles[0]
                else:
                    print(
                        f"    [!] Missing metric profile features for {env_name}/{tier_name}. Skipping row.")
                    continue

            try:
                with open(profile_file, "r") as f:
                    profile = json.load(f)

                # CRITICAL CORRECTION: Navigating nested dictionary architecture from your analyzer script
                coverage_data = profile.get(
                    "coverage", profile.get("Coverage", {}))
                quality_data = profile.get(
                    "quality", profile.get("Quality", {}))
                diversity_data = profile.get(
                    "diversity", profile.get("Diversity", {}))

                # Metric map extractions
                state_coverage = coverage_data.get("State Entropy", np.nan)
                state_spread = coverage_data.get("State Spread", np.nan)
                action_entropy = coverage_data.get("Action Entropy", np.nan)
                reward_sparsity = quality_data.get("Reward Sparsity", 0.0)
                traj_diversity = diversity_data.get(
                    "Trajectory Diversity", np.nan)

                # Check for Minari generation structure indicators
                is_minari = 1 if "v0" in profile.get(
                    "Dataset", "") or "v4" in profile.get("Dataset", "") else 0

                rows.append({
                    "Dataset_ID": f"{env_name}_{tier_name}",
                    "State_Coverage_Entropy": state_coverage,
                    "State_Spread": state_spread,
                    "Action_Entropy": action_entropy,
                    "Trajectory_Diversity": traj_diversity,
                    "Reward_Sparsity": reward_sparsity,
                    "Is_Minari": is_minari,
                    "Normalized_Target_Score": normalized_target
                })
                print(
                    f"    [✓] Aligned: {env_name}/{tier_name} -> Target Score: {normalized_target:.4f}")

            except Exception as e:
                print(
                    f"    [!] Failed compiling row data for {env_name}/{tier_name}: {e}")

    if not rows:
        print("\n[!] Failure: No aligned data rows found.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    print("\n=========================================================")
    print(f"[✓] Success! Meta-Dataset written to: {output_csv}")
    print("=========================================================")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
