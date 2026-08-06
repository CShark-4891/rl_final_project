# train_meta_predictor.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


def main():
    print("=========================================================")
    print("[+] Training Meta-Predictor on Offline RL Data Metrics")
    print("=========================================================")

    # 1. Load the compiled dataset matrix
    df = pd.read_csv("../dataset_profiles/meta_analysis_registry.csv")

    # Define our feature inputs (X) and performance targets (y)
    features = [
        "State_Coverage_Entropy",
        "State_Spread",
        "Action_Entropy",
        "Trajectory_Diversity",
        "Reward_Sparsity",
        "Is_Minari"
    ]

    X = df[features]
    y = df["Normalized_Target_Score"]

    print(f"[+] Loaded {len(df)} dataset profile rows.")
    print(f"[+] Diagnostic Features: {features}\n")

    # 2. Fit a Random Forest Regressor to extract feature importance
    # Using a small estimator tree size due to our compact row count
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    # 3. Print out structural feature importances
    importances = model.feature_importances_

    print("[+] META-MODEL FEATURE IMPORTANCES:")
    for name, importance in zip(features, importances):
        print(f"  • {name:<25}: {importance:.4f}")
    print("=========================================================")

    # 4. Generate a visual importance plot for the final paper write-up
    plt.figure(figsize=(10, 5))
    indices = np.argsort(importances)[::-1]
    plt.title("Impact of Data Metrics on Offline RL Performance (CQL)")
    plt.bar(range(X.shape[1]), importances[indices],
            align="center", color="royalblue")
    plt.xticks(range(X.shape[1]), [features[i]
               for i in indices], rotation=30, ha="right")
    plt.tight_layout()

    plt.savefig("metric_importance_chart.png")
    print("[+] Saved analytical feature importance chart to: metric_importance_chart.png")


if __name__ == "__main__":
    main()
