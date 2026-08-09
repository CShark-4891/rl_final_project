# src/pipeline/predict_performance.py
import os
import json
import glob
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


class OfflineRLMetaPredictor:
    def __init__(self):
        self.features = [
            "State_Coverage_Entropy",
            "State_Spread",
            "Action_Entropy",
            "Trajectory_Diversity"
        ]
        self.model = GradientBoostingRegressor(
            n_estimators=100, max_depth=2, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, registry_csv_path):
        if not os.path.exists(registry_csv_path):
            raise FileNotFoundError(
                f"Registry file not found at {registry_csv_path}")

        df = pd.read_csv(registry_csv_path)
        df_clean = df[df["Normalized_Target_Score"] > -10].copy()

        X = df_clean[self.features]
        y = df_clean["Normalized_Target_Score"]

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        print(
            f"[✓] Meta-Predictor successfully trained on {len(df_clean)} dataset environment tiers.")

    def _extract_features_from_profile(self, profile_json_path):
        """Helper to safely parse profile parameters."""
        with open(profile_json_path, "r") as f:
            profile = json.load(f)

        coverage = profile.get("Coverage", {})
        diversity = profile.get("Diversity", {})

        return pd.DataFrame([{
            "State_Coverage_Entropy": coverage.get("State Entropy", 0.0),
            "State_Spread": coverage.get("State Spread", 0.0),
            "Action_Entropy": coverage.get("Action Entropy", 0.0),
            "Trajectory_Diversity": diversity.get("Trajectory Diversity", 0.0)
        }])

    def predict_dataset(self, profile_json_path):
        """Pass a raw dataset analyzer JSON profile here to predict its CQL score."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before predicting.")
        if not os.path.exists(profile_json_path):
            raise FileNotFoundError(
                f"Profile file missing: {profile_json_path}")

        input_data = self._extract_features_from_profile(profile_json_path)
        input_scaled = self.scaler.transform(input_data[self.features])
        return self.model.predict(input_scaled)[0]

    def cross_validate_with_real_world(self, cql_runs_dir, profiles_dir):
        """
        Crawls the pipeline results, grabs real-world returns, matches them
        with feature profiles, and evaluates the predictor's precision.
        """
        print("\n=========================================================================")
        print("[?] CROSS-EXAMINING META-PREDICTOR VS REAL-WORLD PIPELINE RUNS")
        print("=========================================================================\n")

        # Find all global_pipeline_report.json files down the directory tree
        search_pattern = os.path.join(
            cql_runs_dir, "**", "global_pipeline_report.json")
        report_paths = glob.glob(search_pattern, recursive=True)

        if not report_paths:
            print(
                f"[!] No pipeline execution reports found under {cql_runs_dir}")
            return

        evaluation_rows = []

        for report_path in report_paths:
            with open(report_path, "r") as f:
                report = json.load(f)

            # Extract basic identifiers from the pipeline output
            # e.g., "mujoco/walker2d/simple-v0"
            raw_env_string = report.get("environment", "")
            real_world_score = report.get(
                "aggregate_performance", {}).get("mean_d4rl_score", None)

            if real_world_score is None or not raw_env_string:
                continue

            # Clean identifier syntax to map to the json feature file name
            # e.g., "mujoco/walker2d/simple-v0" -> "mujoco_walker2d_simple-v0.json"
            profile_filename = raw_env_string.replace("/", "_") + ".json"
            # Adjust mapping explicitly for custom variants like medium-replay if directory underscores mismatch
            profile_filename = profile_filename.replace(
                "medium_replay", "medium-replay")

            profile_path = os.path.join(profiles_dir, profile_filename)

            if not os.path.exists(profile_path):
                print(
                    f"[!] Found execution results for {raw_env_string}, but feature profile is missing at {profile_path}. Skipping...")
                continue

            # Get the predictor's opinion
            predicted_score = self.predict_dataset(profile_path)
            abs_error = abs(predicted_score - real_world_score)

            evaluation_rows.append({
                "Environment/Tier": raw_env_string.replace("mujoco/", ""),
                "Real D4RL Score": real_world_score,
                "Predicted Score": predicted_score,
                "Absolute Error": abs_error
            })

        # Convert to DataFrame for pretty structural printing and summaries
        eval_df = pd.DataFrame(evaluation_rows)

        # Separate out visual blocks or mark OOD targets if present
        print(eval_df.to_string(index=False, formatters={
            "Real D4RL Score": "{:.2f}%".format,
            "Predicted Score": "{:.2f}%".format,
            "Absolute Error": "{:.2f}".format
        }))

        mean_absolute_error = eval_df["Absolute Error"].mean()
        print("\n-------------------------------------------------------------------------")
        print("[+] METRIC PERFORMANCE SUMMARY:")
        print(
            f"   Overall Meta-Predictor Mean Absolute Error (MAE): {mean_absolute_error:.4f}")
        print("=========================================================================\n")


if __name__ == "__main__":
    predictor = OfflineRLMetaPredictor()

    # 1. Train on the historical registry
    registry_path = "../dataset_profiles/meta_analysis_registry.csv"
    predictor.train(registry_path)

    # 2. Automatically map out all available pipeline logs against feature targets
    cql_runs_directory = "../results/cql_runs"
    dataset_profiles_directory = "../dataset_profiles"

    predictor.cross_validate_with_real_world(
        cql_runs_dir=cql_runs_directory,
        profiles_dir=dataset_profiles_directory
    )
