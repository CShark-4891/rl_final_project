"""
visualize_results.py
====================
Standalone visualisation script for offline RL experiment results.

Reads pipeline outputs (global_pipeline_report.json), dataset profiles
(mujoco_*.json), and the meta-analysis registry (meta_analysis_registry.csv)
to produce publication-ready figures.

Usage:
    python visualize_results.py
    python visualize_results.py --results-dir ../results/cql_runs
    python visualize_results.py --profiles-dir ../dataset_profiles
    python visualize_results.py --output-dir ../results/figures

    # Plot performance/seed-consistency figures using published d3rlpy paper
    # results (src/results/d3rlpy_paper_results/d4rl.py) instead of our own
    # pipeline results, e.g. to compare against CQL, IQL, TD3+BC, ...
    python visualize_results.py --paper-algorithm IQL

Dependencies: matplotlib, seaborn, pandas, numpy, scikit-learn
No GPU, d3rlpy, torch, or MuJoCo required.
"""

import argparse
import glob
import importlib.util
import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Colour-blind friendly palette (Wong, 2011)
CB_PALETTE = {
    "hopper": "#0072B2",  # blue
    "walker2d": "#D55E00",  # vermillion
    "halfcheetah": "#009E73",  # green
    "ant": "#CC79A7",  # reddish purple
    "humanoid": "#F0E442",  # yellow
    "pendulum": "#56B4E9",  # sky blue
}

ENV_ORDER = ["hopper", "walker2d", "halfcheetah"]
# Includes both our own tiers (simple, expert) and the D4RL v0 tiers used by
# the d3rlpy paper results (random, medium-expert), so the same plots work
# with either data source.
TIER_ORDER = ["simple", "random", "medium",
              "medium-replay", "expert", "medium-expert"]

# Mapping from internal env name to display label
ENV_LABELS = {
    "hopper": "Hopper",
    "walker2d": "Walker2d",
    "halfcheetah": "HalfCheetah",
    "ant": "Ant",
    "humanoid": "Humanoid",
    "pendulum": "Pendulum",
}

TIER_LABELS = {
    "simple": "Simple",
    "random": "Random",
    "medium": "Medium",
    "medium-replay": "Medium-Replay",
    "expert": "Expert",
    "medium-expert": "Medium-Expert",
}

# Features used for meta-predictor and correlation analysis
META_FEATURES = [
    "State_Coverage_Entropy",
    "State_Spread",
    "Action_Entropy",
    "Trajectory_Diversity",
    "Reward_Sparsity",
]

META_FEATURE_LABELS = {
    "State_Coverage_Entropy": "State Coverage Entropy",
    "State_Spread": "State Spread",
    "Action_Entropy": "Action Entropy",
    "Trajectory_Diversity": "Trajectory Diversity",
    "Reward_Sparsity": "Reward Sparsity",
}

# Radar-chart metrics (subset that makes sense on a comparable scale)
# Exclude Reward_Sparsity because it is constant (0.0) for all MuJoCo datasets.
RADAR_METRICS = [
    "State_Coverage_Entropy",
    "State_Spread",
    "Action_Entropy",
    "Trajectory_Diversity",
]

RADAR_LABELS = [
    "State Coverage\nEntropy",
    "State Spread",
    "Action Entropy",
    "Trajectory\nDiversity",
]

# D4RL reference baselines for normalizing raw scores (0% = random, 100% = expert).
# These are used to re-normalize raw scores from pipeline results, ensuring
# correct scores even if the original benchmark run used missing/incorrect refs.
D4RL_REF_SCORES = {
    "halfcheetah": {
        "random": -280.05,
        "expert": 12135.0
    },
    "hopper": {
        "random": -20.0,
        "expert": 3234.3
    },
    "walker2d": {
        "random": 1.62,
        "expert": 4592.3
    },
    "ant": {
        "random": -325.6,
        "expert": 3818.5
    },
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_env_tier_from_path(path: str):
    """Extract (env_name, tier_name) from a path like .../hopper/expert-v0 or .../hopper/expert/..."""
    parts = path.replace("\\", "/").split("/")
    for env in ENV_ORDER:
        if env in parts:
            idx = parts.index(env)
            if idx + 1 < len(parts):
                raw = parts[idx + 1]
                # Strip version suffix like -v0, -v2, -v4, -v5
                tier = raw.split("-")[0] if "-" in raw else raw
                return env, tier
    # Fallback: try to match env name anywhere in the string
    for env in ENV_ORDER:
        if env in path.lower():
            for tier in TIER_ORDER:
                if tier in path.lower():
                    return env, tier
    return None, None


def _load_all_pipeline_reports(results_dir: str) -> pd.DataFrame:
    """Walk results_dir, load every global_pipeline_report.json, return DataFrame."""
    pattern = os.path.join(results_dir, "**", "global_pipeline_report.json")
    paths = glob.glob(pattern, recursive=True)

    rows = []
    for p in paths:
        with open(p, "r") as f:
            data = json.load(f)

        env_name = data.get("environment", "")
        agg = data.get("aggregate_performance", {})
        mean_d4rl = agg.get("mean_d4rl_score")
        std_d4rl = agg.get("std_d4rl_score")

        # Extract env family and tier from the environment string
        env_family, tier = _parse_env_tier_from_path(env_name)
        if env_family is None:
            # Fallback: try from the path itself
            env_family, tier = _parse_env_tier_from_path(p)

        # Skip if we still can't parse env/tier
        if env_family is None or tier is None:
            continue

        # Collect seed-level data AND re-normalize from raw scores
        seed_data = []
        for seed_key, telemetry in data.get("seed_level_telemetry", {}).items():
            raw = telemetry.get("mean_raw_score")
            # Re-normalize from raw score (always correct) rather than trusting
            # the stored d4rl_normalized_score which may have been computed with
            # missing reference values.
            normalized = _normalize_raw_score(
                env_family, raw) if raw is not None else None
            seed_data.append({
                "seed": int(seed_key),
                "mean_raw_score": raw,
                "std_raw_score": telemetry.get("std_raw_score"),
                "d4rl_normalized_score": normalized,
            })

        # Re-compute aggregate from re-normalized seed scores
        valid_scores = [sd["d4rl_normalized_score"]
                        for sd in seed_data if sd["d4rl_normalized_score"] is not None]
        if valid_scores:
            mean_d4rl = float(np.mean(valid_scores))
            std_d4rl = float(np.std(valid_scores))
        else:
            mean_d4rl = None
            std_d4rl = None

        rows.append({
            "env_name": env_name,
            "env_family": env_family,
            "tier": tier,
            "mean_d4rl": mean_d4rl,
            "std_d4rl": std_d4rl,
            "report_path": p,
            "seed_data": seed_data,
        })

    return pd.DataFrame(rows)


def _load_all_dataset_profiles(profiles_dir: str) -> pd.DataFrame:
    """Load all mujoco_*.json profile files into a DataFrame."""
    pattern = os.path.join(profiles_dir, "mujoco_*.json")
    paths = glob.glob(pattern, recursive=False)

    rows = []
    for p in paths:
        with open(p, "r") as f:
            profile = json.load(f)

        dataset_name = profile.get("Dataset", "")
        env_family, tier = _parse_env_tier_from_path(dataset_name)

        coverage = profile.get("Coverage", {})
        quality = profile.get("Quality", {})
        diversity = profile.get("Diversity", {})
        size_info = profile.get("Size", {})

        rows.append({
            "dataset": dataset_name,
            "env_family": env_family,
            "tier": tier,
            "State_Coverage_Entropy": coverage.get("State Entropy", np.nan),
            "State_Spread": coverage.get("State Spread", np.nan),
            "State_Cluster_Coverage": coverage.get("State Cluster Coverage", np.nan),
            "Action_Variance": coverage.get("Action Variance", np.nan),
            "Action_Entropy": coverage.get("Action Entropy", np.nan),
            "Mean_Return": quality.get("Mean Return", np.nan),
            "Std_Return": quality.get("Std Return", np.nan),
            "Reward_Sparsity": quality.get("Reward Sparsity", np.nan),
            "Trajectory_Diversity": diversity.get("Trajectory Diversity", np.nan),
            "Transitions": size_info.get("Transitions", np.nan),
            "Episodes": size_info.get("Episodes", np.nan),
            "Avg_Episode_Length": size_info.get("Average Episode Length", np.nan),
        })

    return pd.DataFrame(rows)


def _load_meta_registry(profiles_dir: str) -> pd.DataFrame:
    """Load the meta_analysis_registry.csv if it exists."""
    csv_path = os.path.join(profiles_dir, "meta_analysis_registry.csv")
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def _load_paper_results() -> dict:
    """Load the RESULTS dict from src/results/d3rlpy_paper_results/d4rl.py.

    Loaded by file path (rather than package import) since this script is
    also run standalone outside the src/ package.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.normpath(
        os.path.join(script_dir, "..", "results",
                     "d3rlpy_paper_results", "d4rl.py")
    )
    spec = importlib.util.spec_from_file_location(
        "d3rlpy_paper_d4rl_results", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RESULTS


def _parse_dataset_id(dataset_id: str):
    """Parse a meta-registry Dataset_ID like 'hopper_medium-replay' into
    (env_family, tier)."""
    if "_" not in dataset_id:
        return None, None
    env_family, tier = dataset_id.split("_", 1)
    return env_family, tier


def _build_meta_with_paper_target(df_meta: pd.DataFrame, algorithm: str) -> pd.DataFrame:
    """Return a copy of df_meta with Normalized_Target_Score replaced by the
    d3rlpy paper's published mean score for `algorithm`, matched per-row via
    Dataset_ID (env_tier -> env-tier-v0).

    The dataset meta-features (state coverage, action entropy, etc.) describe
    properties of our own datasets and stay unchanged — only the performance
    target being explained/predicted is swapped out. Rows whose tier has no
    counterpart in the paper's D4RL v0 naming (our 'simple'/'expert' tiers
    vs. the paper's 'random'/'medium-expert') get NaN and are naturally
    dropped by the existing dropna() calls in each plot function.
    """
    if df_meta.empty:
        return df_meta

    paper_results = _load_paper_results()
    if algorithm not in paper_results:
        available = ", ".join(sorted(paper_results.keys()))
        raise ValueError(
            f"Unknown --paper-algorithm '{algorithm}'. Available: {available}")
    algo_scores = paper_results[algorithm]

    def lookup(dataset_id):
        env_family, tier = _parse_dataset_id(str(dataset_id))
        if env_family is None:
            return np.nan
        scores = algo_scores.get(f"{env_family}-{tier}-v0")
        return scores["mean"] if scores else np.nan

    df = df_meta.copy()
    df["Normalized_Target_Score"] = df["Dataset_ID"].apply(lookup)
    return df


def _parse_paper_dataset_key(dataset_key: str):
    """Parse a d3rlpy paper dataset key like 'halfcheetah-medium-replay-v0'
    into (env_family, tier), mirroring _parse_env_tier_from_path's output."""
    parts = dataset_key.split("-")
    if len(parts) < 3 or not parts[-1].startswith("v"):
        return None, None
    env_family = parts[0]
    tier = "-".join(parts[1:-1])
    if env_family not in ENV_ORDER:
        return None, None
    return env_family, tier


def _load_results_from_paper(algorithm: str) -> pd.DataFrame:
    """Build a DataFrame shaped like _load_all_pipeline_reports()'s output,
    but populated from published d3rlpy paper benchmark results for a given
    algorithm, so it can be fed into the same plotting functions.

    Note: the paper only reports an aggregate mean/std per dataset (no
    per-seed telemetry), so `seed_data` is left empty — plots that rely on
    seed-level detail (e.g. plot_seed_consistency) will simply find nothing
    to plot for this data source.
    """
    paper_results = _load_paper_results()
    if algorithm not in paper_results:
        available = ", ".join(sorted(paper_results.keys()))
        raise ValueError(
            f"Unknown --paper-algorithm '{algorithm}'. Available: {available}")

    rows = []
    for dataset_key, scores in paper_results[algorithm].items():
        env_family, tier = _parse_paper_dataset_key(dataset_key)
        if env_family is None or tier is None:
            continue
        rows.append({
            "env_name": dataset_key,
            "env_family": env_family,
            "tier": tier,
            "mean_d4rl": scores["mean"],
            "std_d4rl": scores["std"],
            "report_path": f"<d3rlpy_paper_results:{algorithm}>",
            "seed_data": [],
        })

    return pd.DataFrame(rows)


def _save_figure(fig, filename: str, output_dir: str, dpi: int = 150):
    """Save a figure to output_dir/filename with consistent settings."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"  [✓] Saved: {path}")
    plt.close(fig)


def _env_color(env_family: str) -> str:
    """Return colour for an environment family."""
    return CB_PALETTE.get(env_family, "#999999")


def _tier_marker(tier: str) -> str:
    """Return marker style for a dataset tier."""
    mapping = {"simple": "o", "medium": "s",
               "medium-replay": "^", "expert": "D"}
    return mapping.get(tier, "o")


# ---------------------------------------------------------------------------
# Plot 1: Performance Overview — grouped bar chart
# ---------------------------------------------------------------------------


def plot_performance_overview(
    df_results: pd.DataFrame,
    output_dir: str,
    algorithm_label: str = "CQL",
    output_filename: str = "performance_overview.png",
):
    """Grouped bar chart: D4RL Normalized Score per (Environment x Tier)."""
    # Filter to rows with valid scores
    df = df_results.dropna(subset=["mean_d4rl"]).copy()
    if df.empty:
        print("  [!] No valid performance data to plot.")
        return

    # Build a pivot table: rows = env_family, columns = tier, values = mean_d4rl
    pivot = df.pivot_table(
        index="env_family",
        columns="tier",
        values="mean_d4rl",
        aggfunc="first",
    )

    # Also get std for error bars
    # std_pivot = df.pivot_table(
    #     index="env_family",
    #     columns="tier",
    #     values="std_d4rl",
    #     aggfunc="first",
    # )

    # Reorder rows and columns
    envs = [e for e in ENV_ORDER if e in pivot.index]
    # tiers = [t for t in TIER_ORDER if t in pivot.columns]

    if not envs:
        print("  [!] No recognised environments in performance data.")
        return

    n_envs = len(envs)
    # n_tiers = len(tiers)
    x = np.arange(n_envs)
    # width = 0.8 / n_tiers

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # for i, tier in enumerate(tiers):
    #     values = [pivot.loc[e, tier]
    #               if tier in pivot.columns else np.nan for e in envs]
    #     errors = [std_pivot.loc[e, tier]
    #               if tier in std_pivot.columns else np.nan for e in envs]
    #     offset = (i - (n_tiers - 1) / 2) * width
    #     bars = ax.bar(
    #         x + offset,
    #         values,
    #         width,
    #         label=TIER_LABELS.get(tier, tier),
    #         color=sns.color_palette("muted")[i % 10],
    #         yerr=errors,
    #         capsize=3,
    #         error_kw={"linewidth": 1},
    #     )

    ax.set_xticks(x)
    ax.set_xticklabels([ENV_LABELS.get(e, e) for e in envs], fontsize=12)
    ax.set_ylabel("D4RL Normalized Score (%)", fontsize=12)
    ax.set_title(
        f"{algorithm_label} Performance Across Environments and Dataset Tiers", fontsize=14)
    ax.axhline(y=0, color="gray", linestyle="--",
               linewidth=0.8, label="Random policy")
    ax.legend(fontsize=10, loc="best")
    ax.grid(axis="y", alpha=0.3)

    _save_figure(fig, output_filename, output_dir)


# ---------------------------------------------------------------------------
# Plot 2: Metrics vs. Performance — scatter plots with trend lines
# ---------------------------------------------------------------------------


def plot_metrics_vs_performance(
    df_meta: pd.DataFrame,
    output_dir: str,
    algorithm_label: str = "CQL",
    output_filename: str = "metrics_vs_performance.png",
):
    """2x2 scatter subplot: each meta-feature vs. D4RL score with trend line."""
    df = df_meta.dropna(subset=META_FEATURES + ["Normalized_Target_Score"]).copy()
    if df.empty:
        print("  [!] No meta-registry data for metrics-vs-performance plot.")
        return

    features = META_FEATURES[:4]  # Use first 4 for 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    for ax, feat in zip(axes.flat, features):
        x = df[feat]
        y = df["Normalized_Target_Score"]

        # Scatter with env colouring
        for env in ENV_ORDER:
            mask = df["Dataset_ID"].str.contains(env, case=False, na=False)
            if mask.any():
                ax.scatter(
                    x[mask], y[mask],
                    c=_env_color(env),
                    label=ENV_LABELS.get(env, env),
                    s=60, alpha=0.8, edgecolors="white", linewidth=0.5,
                    zorder=3,
                )

        # Trend line (polynomial degree 1)
        if len(x) > 2:
            coeffs = np.polyfit(x, y, 1)
            poly = np.poly1d(coeffs)
            x_sorted = np.sort(x)
            ax.plot(x_sorted, poly(x_sorted), color="gray",
                    linestyle="--", linewidth=1.2, zorder=2)

        ax.set_xlabel(META_FEATURE_LABELS.get(feat, feat), fontsize=11)
        ax.set_ylabel("D4RL Normalized Score (%)", fontsize=11)
        ax.grid(alpha=0.3)

    # Single legend for the whole figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(ENV_ORDER),
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(
        f"Dataset Characteristics vs. {algorithm_label} Performance", fontsize=14, y=1.06)
    fig.tight_layout()

    _save_figure(fig, output_filename, output_dir)


# ---------------------------------------------------------------------------
# Plot 3: Correlation Heatmap
# ---------------------------------------------------------------------------


def plot_correlation_heatmap(
    df_meta: pd.DataFrame,
    output_dir: str,
    algorithm_label: str = "CQL",
    output_filename: str = "correlation_heatmap.png",
):
    """Annotated correlation matrix of all numeric meta-features + target."""
    cols = META_FEATURES + ["Normalized_Target_Score"]
    df = df_meta[cols].dropna().copy()
    if df.empty:
        print("  [!] No meta-registry data for correlation heatmap.")
        return

    # Drop constant columns (e.g., Reward_Sparsity=0.0 for all MuJoCo datasets)
    # because they produce NaN correlations and clutter the plot.
    constant_cols = [c for c in cols if df[c].std() == 0]
    if constant_cols:
        print(f"  [!] Dropping constant columns from heatmap: {constant_cols}")
        df = df.drop(columns=constant_cols)

    # Rename columns for readability
    rename_map = {**META_FEATURE_LABELS,
                  "Normalized_Target_Score": "D4RL Score"}
    df_renamed = df.rename(columns=rename_map)

    corr = df_renamed.corr()

    fig, ax = plt.subplots(figsize=(8, 6.5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        vmin=-1, vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.75, "label": "Pearson Correlation"},
        ax=ax,
    )
    ax.set_title(f"Correlation Matrix: Dataset Metrics vs. {algorithm_label} Performance",
                 fontsize=13, pad=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30,
                       ha="right", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    _save_figure(fig, output_filename, output_dir)


# ---------------------------------------------------------------------------
# Plot 4: Feature Importance (Random Forest)
# ---------------------------------------------------------------------------


def plot_feature_importance(
    df_meta: pd.DataFrame,
    output_dir: str,
    algorithm_label: str = "CQL",
    output_filename: str = "feature_importance.png",
):
    """Random Forest feature importance as a horizontal bar chart."""
    df = df_meta.dropna(subset=META_FEATURES + ["Normalized_Target_Score"]).copy()
    if df.empty:
        print("  [!] No meta-registry data for feature importance.")
        return

    X = df[META_FEATURES]
    y = df["Normalized_Target_Score"]

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(META_FEATURES)))[::-1]

    bars = ax.barh(
        range(len(META_FEATURES)),
        importances[indices],
        color=colors,
        align="center",
        edgecolor="white",
        height=0.7,
    )

    ax.set_yticks(range(len(META_FEATURES)))
    ax.set_yticklabels(
        [META_FEATURE_LABELS.get(META_FEATURES[i], META_FEATURES[i])
         for i in indices],
        fontsize=11,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(f"Random Forest Feature Importance\nfor Predicting {algorithm_label} Performance",
                 fontsize=13)
    ax.grid(axis="x", alpha=0.3)

    # Annotate bars with values
    for bar, val in zip(bars, importances[indices]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    _save_figure(fig, output_filename, output_dir)


# ---------------------------------------------------------------------------
# Plot 5: Predicted vs. Actual (Gradient Boosting cross-validation)
# ---------------------------------------------------------------------------


def plot_predicted_vs_actual(
    df_meta: pd.DataFrame,
    output_dir: str,
    algorithm_label: str = "CQL",
    output_filename: str = "predicted_vs_actual.png",
):
    """Scatter plot: predicted D4RL score vs. actual, using a GBR trained on
    the meta-registry with leave-one-dataset-out cross-validation."""

    df = df_meta.dropna(subset=META_FEATURES + ["Normalized_Target_Score"]).copy()
    if len(df) < 3:
        print("  [!] Too few data points for predicted-vs-actual plot.")
        return

    X = df[META_FEATURES].values
    y = df["Normalized_Target_Score"].values
    dataset_ids = df["Dataset_ID"].values

    # Leave-one-out cross-validation at the dataset level
    predictions = np.full_like(y, np.nan)
    scaler = StandardScaler()

    for i in range(len(df)):
        # Hold out one row
        train_mask = np.ones(len(df), dtype=bool)
        train_mask[i] = False

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[~train_mask].reshape(1, -1)

        if len(np.unique(y_train)) < 2:
            predictions[i] = y[i]  # fallback: predict itself
            continue

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = GradientBoostingRegressor(
            n_estimators=50, max_depth=2, random_state=42)
        model.fit(X_train_scaled, y_train)
        predictions[i] = model.predict(X_test_scaled)[0]

    fig, ax = plt.subplots(figsize=(7, 7))

    # Colour by environment
    for env in ENV_ORDER:
        mask = np.array([env in str(did) for did in dataset_ids])
        if mask.any():
            ax.scatter(
                predictions[mask], y[mask],
                c=_env_color(env),
                label=ENV_LABELS.get(env, env),
                s=80, alpha=0.8, edgecolors="white", linewidth=0.7,
                zorder=3,
            )

    # Perfect-prediction line
    lims = [
        min(np.nanmin(predictions), np.nanmin(y)),
        max(np.nanmax(predictions), np.nanmax(y)),
    ]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.6,
            label="Perfect prediction", zorder=1)

    ax.set_xlabel("Predicted D4RL Score (%)", fontsize=12)
    ax.set_ylabel("Actual D4RL Score (%)", fontsize=12)
    ax.set_title(f"Meta-Predictor: Predicted vs. Actual {algorithm_label} Performance\n(Leave-One-Out CV)",
                 fontsize=13)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    # Annotate MAE
    valid = ~np.isnan(predictions)
    mae = np.mean(np.abs(predictions[valid] - y[valid]))
    ax.text(0.95, 0.05, f"MAE = {mae:.2f}%", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    _save_figure(fig, output_filename, output_dir)


# ---------------------------------------------------------------------------
# Plot 6: Seed Consistency — boxplot of seed-level D4RL scores
# ---------------------------------------------------------------------------


def plot_seed_consistency(
    df_results: pd.DataFrame,
    output_dir: str,
    output_filename: str = "seed_consistency.png",
):
    """Boxplot showing the distribution of seed-level D4RL scores per dataset."""
    # Build seed-level DataFrame
    rows = []
    for _, row in df_results.iterrows():
        env_family = row["env_family"]
        tier = row["tier"]
        if env_family is None or tier is None:
            continue
        if pd.isna(env_family) or pd.isna(tier):
            continue
        for sd in row.get("seed_data", []):
            score = sd.get("d4rl_normalized_score")
            if score is not None:
                rows.append({
                    "dataset": f"{ENV_LABELS.get(env_family, env_family)}-{TIER_LABELS.get(tier, tier)}",
                    "env_family": env_family,
                    "tier": tier,
                    "score": score,
                })

    df_seeds = pd.DataFrame(rows)
    if df_seeds.empty:
        print("  [!] No seed-level data for consistency plot.")
        return

    # Sort datasets for consistent display
    datasets_order = sorted(df_seeds["dataset"].unique())

    fig, ax = plt.subplots(figsize=(12, 5.5))

    # Colour boxes by environment
    palette = {}
    for ds in datasets_order:
        for env in ENV_ORDER:
            if env in ds.lower():
                palette[ds] = _env_color(env)
                break
        else:
            palette[ds] = "#999999"

    # Use hue instead of palette to avoid deprecation warning
    df_seeds["env_for_hue"] = df_seeds["env_family"].map(
        lambda e: ENV_LABELS.get(e, e))

    sns.boxplot(
        data=df_seeds,
        x="dataset",
        y="score",
        hue="env_for_hue",
        palette={ENV_LABELS.get(e, e): _env_color(e) for e in ENV_ORDER},
        width=0.6,
        linewidth=1.2,
        legend=False,
        ax=ax,
    )

    # Overlay individual seed points
    sns.stripplot(
        data=df_seeds,
        x="dataset",
        y="score",
        color="black",
        size=5,
        alpha=0.6,
        jitter=True,
        ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel("D4RL Normalized Score (%)", fontsize=12)
    ax.set_title("Seed-Level Consistency Across Datasets", fontsize=14)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35,
                       ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    _save_figure(fig, output_filename, output_dir)


# ---------------------------------------------------------------------------
# Plot 7: Radar Chart — Expert vs. Simple per Environment
# ---------------------------------------------------------------------------


def plot_radar_comparison(df_profiles: pd.DataFrame, output_dir: str):
    """Radar chart comparing Expert vs. Simple dataset profiles per environment."""
    df = df_profiles.dropna(subset=RADAR_METRICS).copy()
    if df.empty:
        print("  [!] No profile data for radar comparison.")
        return

    # For each environment, get expert and simple tiers
    n_metrics = len(RADAR_METRICS)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    for env in ENV_ORDER:
        env_df = df[df["env_family"] == env]
        expert = env_df[env_df["tier"] == "expert"]
        simple = env_df[env_df["tier"] == "simple"]

        if expert.empty or simple.empty:
            continue

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})

        for label, row in [("Expert", expert.iloc[0]), ("Simple", simple.iloc[0])]:
            # values = [row[m] for m in RADAR_METRICS]
            # Normalise to [0, 1] using min-max across all profiles for this metric
            values_norm = []
            for m in RADAR_METRICS:
                vmin = df[m].min()
                vmax = df[m].max()
                if vmax - vmin > 1e-10:
                    values_norm.append((row[m] - vmin) / (vmax - vmin))
                else:
                    values_norm.append(0.5)
            values_norm += values_norm[:1]  # Close the loop

            color = "#D55E00" if label == "Expert" else "#0072B2"
            ax.fill(angles, values_norm, alpha=0.1, color=color)
            ax.plot(angles, values_norm, "o-",
                    linewidth=2, label=label, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(RADAR_LABELS, fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7)
        ax.set_title(f"{ENV_LABELS.get(env, env)}: Expert vs. Simple Dataset Profile",
                     fontsize=13, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

        _save_figure(fig, f"radar_{env}_expert_vs_simple.png", output_dir)


# ---------------------------------------------------------------------------
# Plot 8: Radar Chart — All datasets in one plot
# ---------------------------------------------------------------------------


def plot_all_datasets_radar(df_profiles: pd.DataFrame, output_dir: str):
    """One radar chart per environment, with each tier (simple/medium/expert)
    shown as a separate coloured polygon."""
    df = df_profiles.dropna(subset=RADAR_METRICS).copy()
    df = df[df["env_family"].notna() & df["tier"].notna()]
    df = df[df["env_family"].apply(lambda x: x is not None)]
    df = df[df["tier"].apply(lambda x: x is not None)]
    if df.empty:
        print("  [!] No profile data for per-environment radar charts.")
        return

    n_metrics = len(RADAR_METRICS)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    # Tier-specific colours (sequential palette: simple→medium→expert)
    tier_colors = {
        "simple": "#fee5d9",  # light orange
        "medium": "#fcae91",  # medium orange
        "medium-replay": "#fb6a4a",  # darker orange
        "expert": "#de2d26",  # red
    }
    tier_order = ["simple", "medium", "medium-replay", "expert"]

    for env in ENV_ORDER:
        env_df = df[df["env_family"] == env]
        if env_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

        for tier in tier_order:
            tier_rows = env_df[env_df["tier"] == tier]
            if tier_rows.empty:
                continue

            for _, row in tier_rows.iterrows():
                # values = [row[m] for m in RADAR_METRICS]
                # Normalise using global min-max (across ALL datasets)
                values_norm = []
                for m in RADAR_METRICS:
                    vmin = df[m].min()
                    vmax = df[m].max()
                    if vmax - vmin > 1e-10:
                        values_norm.append((row[m] - vmin) / (vmax - vmin))
                    else:
                        values_norm.append(0.5)
                values_norm += values_norm[:1]

                color = tier_colors.get(tier, "#999999")
                label = TIER_LABELS.get(tier, tier)

                ax.fill(angles, values_norm, alpha=0.25, color=color)
                ax.plot(angles, values_norm, "o-", linewidth=2,
                        label=label, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(RADAR_LABELS, fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7)
        ax.set_title(f"{ENV_LABELS.get(env, env)}: Dataset Profile by Tier",
                     fontsize=14, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1),
                  fontsize=10, frameon=True)

        _save_figure(fig, f"radar_{env}_all_tiers.png", output_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualisations for offline RL experiment results.")
    parser.add_argument(
        "--results-dir", type=str, default="../policy_results/self_trained_cql_results",
        help="Path to the cql_runs results directory.")
    parser.add_argument(
        "--profiles-dir", type=str, default="../dataset_profiles",
        help="Path to the dataset_profiles directory.")
    parser.add_argument(
        "--output-dir", type=str, default="../results/figures",
        help="Directory to save generated figures.")
    parser.add_argument(
        "--paper-algorithm", type=str, default=None,
        help="If set, plot performance/seed-consistency figures using published "
             "d3rlpy paper D4RL results for this algorithm (see "
             "src/results/d3rlpy_paper_results/d4rl.py for available names, e.g. "
             "CQL, IQL, TD3+BC) instead of our own pipeline results.")
    args = parser.parse_args()

    # Resolve paths relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.normpath(os.path.join(script_dir, args.results_dir))
    profiles_dir = os.path.normpath(
        os.path.join(script_dir, args.profiles_dir))
    output_dir = os.path.normpath(os.path.join(script_dir, args.output_dir))

    print("=" * 60)
    print("  Offline RL Results Visualisation")
    print("=" * 60)
    print(f"  Results dir:  {results_dir}")
    print(f"  Profiles dir: {profiles_dir}")
    print(f"  Output dir:   {output_dir}")
    print("=" * 60)

    # ---- Load data ----
    print("\n[1/3] Loading pipeline results...")
    if args.paper_algorithm:
        print(
            f"       Using d3rlpy paper results for algorithm: {args.paper_algorithm}")
        df_results = _load_results_from_paper(args.paper_algorithm)
    else:
        df_results = _load_all_pipeline_reports(results_dir)
    print(f"       Found {len(df_results)} dataset report(s).")

    print("\n[2/3] Loading dataset profiles...")
    df_profiles = _load_all_dataset_profiles(profiles_dir)
    print(f"       Found {len(df_profiles)} profile(s).")

    print("\n[3/3] Loading meta-analysis registry...")
    df_meta = _load_meta_registry(profiles_dir)
    print(f"       Found {len(df_meta)} row(s).")

    # When plotting against paper results, swap in the paper algorithm's
    # scores as the prediction target while keeping our own dataset
    # meta-features. Rows whose tier has no counterpart in the paper's D4RL
    # naming (our 'simple'/'expert' vs. the paper's 'random'/'medium-expert')
    # end up with no target and are dropped by each plot function as usual.
    algo_label = "CQL"
    filename_suffix = ""
    if args.paper_algorithm:
        algo_label = args.paper_algorithm
        filename_suffix = "_" + args.paper_algorithm.lower().replace("+", "plus")
        df_meta = _build_meta_with_paper_target(df_meta, args.paper_algorithm)
        n_matched = df_meta["Normalized_Target_Score"].notna(
        ).sum() if not df_meta.empty else 0
        print(f"       Matched {n_matched}/{len(df_meta)} row(s) to '{algo_label}' paper scores "
              f"(tier-name mismatches, e.g. simple/expert vs. random/medium-expert, are dropped).")

    # ---- Generate plots ----
    print("\n" + "=" * 60)
    print("  Generating Figures")
    print("=" * 60)

    print("\n  • Performance Overview (grouped bar chart)...")
    plot_performance_overview(
        df_results, output_dir,
        algorithm_label=algo_label,
        output_filename=f"performance_overview{filename_suffix}.png",
    )

    print("\n  • Metrics vs. Performance (scatter + trend)...")
    plot_metrics_vs_performance(
        df_meta, output_dir,
        algorithm_label=algo_label,
        output_filename=f"metrics_vs_performance{filename_suffix}.png",
    )

    print("\n  • Correlation Heatmap...")
    plot_correlation_heatmap(
        df_meta, output_dir,
        algorithm_label=algo_label,
        output_filename=f"correlation_heatmap{filename_suffix}.png",
    )

    print("\n  • Feature Importance (Random Forest)...")
    plot_feature_importance(
        df_meta, output_dir,
        algorithm_label=algo_label,
        output_filename=f"feature_importance{filename_suffix}.png",
    )

    print("\n  • Predicted vs. Actual (LOO-CV)...")
    plot_predicted_vs_actual(
        df_meta, output_dir,
        algorithm_label=algo_label,
        output_filename=f"predicted_vs_actual{filename_suffix}.png",
    )

    print("\n  • Seed Consistency (boxplot)...")
    plot_seed_consistency(
        df_results, output_dir,
        output_filename=f"seed_consistency{filename_suffix}.png",
    )

    print("\n  • Radar: Expert vs. Simple per Environment...")
    plot_radar_comparison(df_profiles, output_dir)

    print("\n  • Radar: All Tiers per Environment...")
    plot_all_datasets_radar(df_profiles, output_dir)

    print("\n" + "=" * 60)
    print(f"  All figures saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
