from configs import default_paths
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from functools import reduce
import json
import os
import sys
import warnings

# Windows consoles default to a legacy codepage (e.g. cp1252) that can't
# encode the "✓"/"•" characters used in status prints below; force UTF-8 so
# the script doesn't crash mid-run on a stock Windows terminal.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
# headless: this script only saves figures, never shows them
matplotlib.use("Agg")


warnings.filterwarnings("ignore", category=UserWarning)

# Project root, two levels up from this script (src/visualization/... -> src
# -> project root). default_paths.py constants are expressed relative to it.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))

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

# Canonical difficulty-rank groups used to line up d4rl's tier naming
# (random/medium/medium-replay/medium-expert) against ours
# (simple/medium/expert) for cross-dataset-source comparisons: simple and
# random are both the "weakest" tier, expert and medium-expert are both the
# "strongest" tier, medium matches exactly, and medium-replay has no Minari
# counterpart at all (rows for it only ever show the d4rl bar).
TIER_GROUP_ORDER = ["weakest", "medium", "medium-replay", "strongest"]
TIER_GROUP_LABELS = {
    "weakest": "Simple / Random",
    "medium": "Medium",
    "medium-replay": "Medium-Replay",
    "strongest": "Expert / Medium-Expert",
}
TIER_TO_GROUP = {
    "simple": "weakest",
    "random": "weakest",
    "medium": "medium",
    "medium-replay": "medium-replay",
    "expert": "strongest",
    "medium-expert": "strongest",
}

# Maps a policy-result source (top-level key under POLICY_RESULTS_DIR, e.g.
# "self_trained") to the dataset-profile source (top-level key under
# DATASET_PROFILES_DIR, e.g. "minari") whose datasets it was trained/
# evaluated on, so score-dependent plots can be joined against the right
# meta-features. Not derivable from the two directory trees themselves since
# their top-level names don't match ("self_trained" vs "minari", etc.).
RESULT_SOURCE_TO_PROFILE_SOURCE = {
    "self_trained": "minari",   # our own CQL runs, trained on Minari-formatted datasets
    "d3rlpy_paper": "d4rl",     # published d3rlpy benchmark numbers on D4RL v0 datasets
}

# Dataset-profile feature vocabulary. Every feature used anywhere below is
# declared once here as membership in exactly one of three lists, mirroring
# the profile JSON's sections (see profile_feature_computer.py /
# dataset_analyzer.py): Coverage, Quality, and Size ("dataset shape").
# FEATURE_LABELS is the single label lookup shared by all of them.
#
# plot_dataset_source_comparison draws Coverage and Quality as one shared-axis
# bar panel each, since every feature within each list is roughly comparable
# scale. Size is different: Transitions/Episodes/Avg_Episode_Length are
# related (Transitions ~= Episodes * Avg_Episode_Length) but differ from each
# other by orders of magnitude, so cramming them onto one shared axis would
# make the smaller ones invisible. Size instead gets one independently-scaled
# sub-panel per metric (see the Size-panel loop in plot_dataset_source_comparison).

FEATURE_LABELS = {
    # Coverage
    "State_Coverage_Entropy": "State Coverage Entropy",
    "State_Standard_Deviation": "State Standard Deviation",
    "State_Cluster_Coverage": "State Cluster Coverage",
    "Action_Standard_Deviation": "Action Standard Deviation",
    "Action_Entropy": "Action Usage Entropy",
    "Trajectory_Diversity": "Trajectory Diversity",
    "EAS": "Mean Expected Action Stochasticity (EAS)",
    "SACo": "State-Action Coverage (SACo)",
    # Quality
    "Reward_Sparsity": "Reward Sparsity",
    "ERI": "Expected Relative Return Improvement (ERI)",
    "TQ": "Trajectory Quality (TQ)",
    # Size / dataset shape
    "Mean_Return": "Mean Return",
    "Transitions": "Total Transitions",
    "Episodes": "Episodes",
    "Avg_Episode_Length": "Avg. Episode Length",
}

# List 1: Coverage features — how much of the state-action space / behavioral
# diversity the dataset exhibits. Comparable [0, ~few]-scale; also doubles as
# the radar-chart axes (both radar functions min-max normalize per metric
# regardless of scale, so scale-comparability only matters for the bar
# panel).
COVERAGE_FEATURES = [
    "State_Standard_Deviation",
    "Action_Standard_Deviation",
    "State_Cluster_Coverage",
    "State_Coverage_Entropy",
    "Action_Entropy",
    "Trajectory_Diversity",
    "EAS",
    "SACo",
]

# List 2: Quality features — how good the dataset's trajectories are.
# Bounded/ratio scale (Reward_Sparsity, TQ in [0, 1]; ERI in [-1, 0]).
QUALITY_FEATURES = [
    "ERI",
    "TQ",
    "Reward_Sparsity"
]

# List 3: Size / "dataset shape" features — absolute-scale counts. Total
# Transitions is split into Episodes and their Avg_Episode_Length
# (Transitions ~= Episodes * Avg_Episode_Length), so each gets its own
# independently-scaled sub-panel in plot_dataset_source_comparison rather
# than sharing one axis (see the module comment above).
SIZE_FEATURES = [
    "Transitions",
    "Episodes",
    "Avg_Episode_Length",
]

# Meta-predictor / correlation feature set: every Coverage + Quality
# feature. Size features are excluded — they describe how much data there
# is, not the characteristics of how it was collected.
META_FEATURES = COVERAGE_FEATURES + QUALITY_FEATURES

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

OUTPUT_FORMAT = ".svg"


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


def _parse_env_tier_from_path(dataset_name: str):
    """Extract (env_name, tier_name) from a path like .../hopper/expert-v0 or .../hopper/expert/..."""
    parts = dataset_name.replace("\\", "/").replace("/", "-").split("-")
    for env in ENV_ORDER:
        if env in parts:
            idx = parts.index(env)
            if idx + 1 < len(parts):
                tier = reduce(lambda x, y: x + "-" + y, parts[idx + 1:])
                return env, tier
        else:
            print(
                f"Warning: Environment '{env}' not found considered for visualization.")
    return None, None


def _load_all_policy_reports(policy_results_dir: str = default_paths.POLICY_RESULTS_DIR) -> dict:
    """Walk results_dir, load every global_pipeline_report.json, return DataFrame."""
    result_sources = os.listdir(policy_results_dir)

    ret = {}

    for result_source in result_sources:
        if result_source not in ret.keys():
            ret[result_source] = {}

        if not os.path.isdir(os.path.join(policy_results_dir, result_source)):
            continue

        source_path = os.path.join(policy_results_dir, result_source)

        files_and_dirs = os.listdir(source_path)

        # check wehther there is only one json object
        json_objects = [f for f in files_and_dirs if f.endswith(".json")]
        if len(json_objects) != 1:
            raise Exception(
                f"Expected exactly one combined JSON score object in {source_path}, found {len(json_objects)}")

        with open(os.path.join(source_path, json_objects[0]), "r") as f:
            data = json.load(f)

            ret[result_source] = data

    return ret


def _load_all_dataset_profiles(profiles_dir: str = default_paths.DATASET_PROFILES_DIR) -> dict:
    """Load all mujoco_*.json profile files into a DataFrame."""
    sources = os.listdir(profiles_dir)

    ret = {}
    for source in sources:
        if source not in ret.keys():
            ret[source] = {}
        source_path = os.path.join(profiles_dir, source)

        if not os.path.isdir(source_path):
            continue

        profiles = os.listdir(source_path)

        for profile in profiles:
            if not profile.endswith(".json"):
                continue

            profile_path = os.path.join(source_path, profile)
            with open(profile_path, "r") as f:
                data = json.load(f)

            dataset_name = data.get("Dataset", "")
            env_family, tier = _parse_env_tier_from_path(dataset_name)
            if not env_family or not tier:
                continue
            if env_family not in ret[source].keys():
                ret[source][env_family] = {}

            coverage = data.get("Coverage", {})
            quality = data.get("Quality", {})
            diversity = data.get("Diversity", {})
            size_info = data.get("Size", {})

            ret[source][env_family][tier] = {
                "dataset": dataset_name,
                "env_family": env_family,
                "tier": tier,
                "State_Coverage_Entropy": coverage.get("State Cluster Entropy", np.nan),
                "State_Standard_Deviation": coverage.get("State Standard Deviation", np.nan),
                "State_Cluster_Coverage": coverage.get("State Cluster Coverage", np.nan),
                "Action_Standard_Deviation": coverage.get("Action Standard Deviation", np.nan),
                "Action_Entropy": coverage.get("Action Usage Entropy", np.nan),
                "EAS": coverage.get("EAS", np.nan),
                "SACo": coverage.get("SACo", np.nan),
                "Mean_Return": quality.get("Mean Return", np.nan),
                "Std_Return": quality.get("Std Return", np.nan),
                "Reward_Sparsity": quality.get("Reward Sparsity", np.nan),
                "ERI": quality.get("ERI", np.nan),
                "TQ": quality.get("TQ", np.nan),
                "Trajectory_Diversity": diversity.get("Trajectory Diversity", np.nan),
                "Transitions": size_info.get("Transitions", np.nan),
                "Episodes": size_info.get("Episodes", np.nan),
                "Avg_Episode_Length": size_info.get("Average Episode Length", np.nan),
            }

    return ret


def _normalize_tier(tier_str: str) -> str:
    """Normalize a raw tier string into our plain tier vocabulary
    ("simple", "expert", "medium-replay", ...) used by TIER_ORDER/TIER_LABELS
    and the tier-equality checks in the radar plots.

    Dataset profiles carry tiers straight from D4RL/Minari-style dataset
    names (e.g. "simple-v0", "medium-replay-v0"), and the combined score
    JSONs are hand/script-assembled and mix the same "-v0" suffix, a stray
    trailing space (e.g. "simple-v0 "), and bare tiers with no suffix at all
    (e.g. pendulum's "replay"). Stripping the suffix from both sides here is
    what lets a profile's "simple-v0" line up with a score's "simple-v0 ".
    """
    tier = tier_str.strip()
    if tier.endswith("-v0"):
        tier = tier[:-len("-v0")]
    return tier


def _flatten_profile_rows(source_profiles: dict) -> list:
    """Flatten one dataset_profile_data[<dataset source>] entry (env_family ->
    tier -> feature dict, as built by _load_all_dataset_profiles) into a flat
    list of per-(env, tier) row dicts. Shared base for _profiles_dict_to_df()
    and _all_profiles_dict_to_df()."""
    rows = []
    for env_family, tiers in source_profiles.items():
        for raw_tier, features in tiers.items():
            tier = _normalize_tier(raw_tier)
            row = dict(features)
            row["tier"] = tier
            row["Dataset_ID"] = f"{env_family}_{tier}"
            rows.append(row)
    return rows


def _profiles_dict_to_df(source_profiles: dict) -> pd.DataFrame:
    """Flatten one dataset_profile_data[<dataset source>] entry into the flat
    per-(env, tier) row DataFrame the profile-driven plot functions expect."""
    columns = ["env_family", "tier", "Dataset_ID"] + META_FEATURES
    rows = _flatten_profile_rows(source_profiles)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)


def _all_profiles_dict_to_df(dataset_profile_data: dict) -> pd.DataFrame:
    """Flatten the full dataset_profile_data dict (dataset_source ->
    env_family -> tier -> feature dict) into one combined per-(source, env,
    tier) row DataFrame spanning every dataset source, for cross-source
    comparison plots."""
    columns = ["source", "env_family", "tier", "Dataset_ID"] + META_FEATURES
    rows = []
    for source, profiles in dataset_profile_data.items():
        for row in _flatten_profile_rows(profiles):
            row = dict(row)
            row["source"] = source
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)


def _all_scores_dict_to_df(source_scores: dict) -> pd.DataFrame:
    """Flatten one policy_score_data[<result source>] entry (algorithm ->
    env_family -> tier_key -> {mean, std}, as built by
    _load_all_policy_reports) into a flat per-(algorithm, env, tier) row
    DataFrame spanning every algorithm in that result source. Used directly
    by the cross-algorithm comparison plots, and as the shared base that
    _scores_dict_to_df() filters down to a single algorithm."""
    columns = ["algorithm", "env_family", "tier",
               "Dataset_ID", "mean_d4rl", "std_d4rl"]
    rows = []
    for algorithm, envs in source_scores.items():
        for env_family, tiers in envs.items():
            for tier_key, scores in tiers.items():
                tier = _normalize_tier(tier_key)
                rows.append({
                    "algorithm": algorithm,
                    "env_family": env_family,
                    "tier": tier,
                    "Dataset_ID": f"{env_family}_{tier}",
                    "mean_d4rl": scores.get("mean"),
                    "std_d4rl": scores.get("std"),
                })
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)


def _scores_dict_to_df(source_scores: dict, algorithm: str) -> pd.DataFrame:
    """Filter _all_scores_dict_to_df() down to one algorithm, in the flat
    per-(env, tier) row shape the performance-driven plot functions expect.

    The combined score files only carry the mean/std aggregate per dataset
    (no per-seed telemetry), so `seed_data` is always empty here — plots that
    rely on seed-level detail (plot_seed_consistency) simply find nothing to
    plot, matching how sparse/aggregate-only sources have always degraded.
    """
    columns = ["env_family", "tier", "Dataset_ID",
               "mean_d4rl", "std_d4rl", "seed_data"]
    df = _all_scores_dict_to_df(source_scores)
    df = df[df["algorithm"] == algorithm].drop(columns=["algorithm"])
    if df.empty:
        return pd.DataFrame(columns=columns)
    df = df.reset_index(drop=True)
    df["seed_data"] = [[] for _ in range(len(df))]
    return df


def _build_meta_df(df_profiles: pd.DataFrame, df_results: pd.DataFrame) -> pd.DataFrame:
    """Join dataset meta-features with policy scores on (env_family, tier) to
    build the combined table the meta-predictor / correlation plots need.

    Replaces the old meta_analysis_registry.csv: that registry is no longer
    produced by the current data pipeline, but its shape (Dataset_ID +
    META_FEATURES + Normalized_Target_Score) is reconstructed here directly
    from the two dicts _load_all_dataset_profiles/_load_all_policy_reports
    already return.
    """
    columns = ["Dataset_ID"] + META_FEATURES + ["Normalized_Target_Score"]
    if df_profiles.empty or df_results.empty:
        return pd.DataFrame(columns=columns)
    merged = df_profiles.merge(
        df_results[["env_family", "tier", "mean_d4rl"]],
        on=["env_family", "tier"],
        how="inner",
    )
    return merged.rename(columns={"mean_d4rl": "Normalized_Target_Score"})


def _slugify_algorithm(algorithm: str) -> str:
    """Turn an algorithm name like 'TD3+BC' into a directory-safe slug."""
    return algorithm.lower().replace("+", "plus").replace(" ", "_")


def _wrap_label(label: str) -> str:
    """Break a tick label into one line per word, so a rotated multi-word
    label (e.g. 'State Coverage Entropy') stays compact instead of one long
    diagonal string."""
    return label.replace(" ", "\n")


def _format_size_value(value: float, feature: str) -> str:
    """Plain, non-scientific text for a Size-panel bar/tick: comma-grouped
    whole number for the two count features (Transitions/Episodes), one
    decimal place for the averaged Avg_Episode_Length. Used for both the
    per-bar value labels and the y-axis tick labels so the panel never falls
    back to matplotlib's default scientific/offset notation for large
    counts — these are exact quantities, not floating-point measurements."""
    if pd.isna(value):
        return ""
    if feature == "Avg_Episode_Length":
        return f"{value:,.1f}"
    return f"{value:,.0f}"


def _usable_features(df: pd.DataFrame, features: list) -> list:
    """Return the subset of `features` that have at least one non-null value
    in `df`. A feature that's entirely missing for a data source (e.g.
    EAS/ERI not yet computed for the D4RL profiles, which predate those
    fields) degrades to "excluded from this plot" instead of, via a blanket
    dropna(subset=features), wiping out every row just because one column is
    always NaN."""
    return [f for f in features if f in df.columns and df[f].notna().any()]


def _save_figure(fig, filename: str, output_dir: str, dpi: int = 150):
    """Save a figure to output_dir/filename with consistent settings."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    _, ext = os.path.splitext(path)
    if OUTPUT_FORMAT != ext:
        path = path.replace(ext, OUTPUT_FORMAT)

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

    # Build pivot tables: rows = env_family, columns = tier, values = mean/std
    pivot = df.pivot_table(
        index="env_family",
        columns="tier",
        values="mean_d4rl",
        aggfunc="first",
    )
    std_pivot = df.pivot_table(
        index="env_family",
        columns="tier",
        values="std_d4rl",
        aggfunc="first",
    )

    # Reorder rows and columns
    envs = [e for e in ENV_ORDER if e in pivot.index]
    tiers = [t for t in TIER_ORDER if t in pivot.columns]

    if not envs:
        print("  [!] No recognised environments in performance data.")
        return
    if not tiers:
        print("  [!] No recognised dataset tiers in performance data.")
        return

    n_envs = len(envs)
    n_tiers = len(tiers)
    x = np.arange(n_envs)
    width = 0.8 / n_tiers

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, tier in enumerate(tiers):
        values = [pivot.loc[e, tier] for e in envs]
        errors = [std_pivot.loc[e, tier]
                  if tier in std_pivot.columns else np.nan for e in envs]
        offset = (i - (n_tiers - 1) / 2) * width
        ax.bar(
            x + offset,
            values,
            width,
            label=TIER_LABELS.get(tier, tier),
            color=sns.color_palette("muted")[i % 10],
            yerr=errors,
            capsize=3,
            error_kw={"linewidth": 1},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([ENV_LABELS.get(e, e) for e in envs], fontsize=10)
    ax.set_ylabel("D4RL Normalized Score (%)", fontsize=10)
    ax.set_title(
        f"{algorithm_label} Performance Across Environments and Dataset Tiers", fontsize=10)
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
    """Grid of scatter subplots, one per meta-feature vs. D4RL score with trend line."""
    features = _usable_features(df_meta, META_FEATURES)
    df = df_meta.dropna(subset=features + ["Normalized_Target_Score"]).copy()
    if df.empty or not features:
        print("  [!] No meta-registry data for metrics-vs-performance plot.")
        return

    n_cols = 2
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(11, 4.5 * n_rows), squeeze=False)

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

        # Trend line (polynomial degree 1). Skip when x is constant (e.g.
        # Reward_Sparsity is 0.0 for every MuJoCo dataset) since a
        # zero-variance x makes the least-squares fit singular and
        # np.polyfit raises LinAlgError instead of returning a flat line.
        if len(x) > 2 and x.std() > 0:
            coeffs = np.polyfit(x, y, 1)
            poly = np.poly1d(coeffs)
            x_sorted = np.sort(x)
            ax.plot(x_sorted, poly(x_sorted), color="gray",
                    linestyle="--", linewidth=1.2, zorder=2)

        ax.set_xlabel(FEATURE_LABELS.get(feat, feat), fontsize=10)
        ax.set_ylabel("D4RL Normalized Score (%)", fontsize=10)
        ax.grid(alpha=0.3)

    # Hide any unused subplot slots when len(features) doesn't fill the grid
    for empty_ax in axes.flat[len(features):]:
        empty_ax.axis("off")

    # Single legend for the whole figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(ENV_ORDER),
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(
        f"Dataset Characteristics vs. {algorithm_label} Performance", fontsize=10, y=1.06)
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
    features = _usable_features(df_meta, META_FEATURES)
    cols = features + ["Normalized_Target_Score"]
    df = df_meta[cols].dropna().copy()
    if df.empty or not features:
        print("  [!] No meta-registry data for correlation heatmap.")
        return

    # Drop constant columns (e.g., Reward_Sparsity=0.0 for all MuJoCo datasets)
    # because they produce NaN correlations and clutter the plot.
    constant_cols = [c for c in cols if df[c].std() == 0]
    if constant_cols:
        print(f"  [!] Dropping constant columns from heatmap: {constant_cols}")
        df = df.drop(columns=constant_cols)

    # Rename columns for readability
    rename_map = {**FEATURE_LABELS,
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
                 fontsize=10, pad=15)
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
    features = _usable_features(df_meta, META_FEATURES)
    df = df_meta.dropna(subset=features + ["Normalized_Target_Score"]).copy()
    if df.empty or not features:
        print("  [!] No meta-registry data for feature importance.")
        return

    X = df[features]
    y = df["Normalized_Target_Score"]

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(features)))[::-1]

    bars = ax.barh(
        range(len(features)),
        importances[indices],
        color=colors,
        align="center",
        edgecolor="white",
        height=0.7,
    )

    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(
        [FEATURE_LABELS.get(features[i], features[i])
         for i in indices],
        fontsize=10,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance", fontsize=10)
    ax.set_title(f"Random Forest Feature Importance\nfor Predicting {algorithm_label} Performance",
                 fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    # Annotate bars with values
    for bar, val in zip(bars, importances[indices]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)

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

    features = _usable_features(df_meta, META_FEATURES)
    df = df_meta.dropna(subset=features + ["Normalized_Target_Score"]).copy()
    if len(df) < 3 or not features:
        print("  [!] Too few data points for predicted-vs-actual plot.")
        return

    X = df[features].values
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

    ax.set_xlabel("Predicted D4RL Score (%)", fontsize=10)
    ax.set_ylabel("Actual D4RL Score (%)", fontsize=10)
    ax.set_title(f"Meta-Predictor: Predicted vs. Actual {algorithm_label} Performance\n(Leave-One-Out CV)",
                 fontsize=10)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    # Annotate MAE
    valid = ~np.isnan(predictions)
    mae = np.mean(np.abs(predictions[valid] - y[valid]))
    ax.text(0.95, 0.05, f"MAE = {mae:.2f}%", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10,
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
    ax.set_ylabel("D4RL Normalized Score (%)", fontsize=10)
    ax.set_title("Seed-Level Consistency Across Datasets", fontsize=10)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35,
                       ha="right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    _save_figure(fig, output_filename, output_dir)


# ---------------------------------------------------------------------------
# Plot 7: Radar Chart — Expert vs. Simple per Environment
# ---------------------------------------------------------------------------


def plot_radar_comparison(df_profiles: pd.DataFrame, output_dir: str):
    """Radar chart comparing Expert vs. Simple dataset profiles per environment."""
    metrics = _usable_features(df_profiles, COVERAGE_FEATURES)
    df = df_profiles.dropna(subset=metrics).copy()
    if df.empty or not metrics:
        print("  [!] No profile data for radar comparison.")
        return

    # For each environment, get expert and simple tiers
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    radar_labels = [_wrap_label(FEATURE_LABELS.get(m, m))
                    for m in metrics]

    for env in ENV_ORDER:
        env_df = df[df["env_family"] == env]
        expert = env_df[env_df["tier"] == "expert"]
        simple = env_df[env_df["tier"] == "simple"]

        if expert.empty or simple.empty:
            continue

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})

        for label, row in [("Expert", expert.iloc[0]), ("Simple", simple.iloc[0])]:
            # Normalise to [0, 1] using min-max across all profiles for this metric
            values_norm = []
            for m in metrics:
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
        ax.set_xticklabels(radar_labels, fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=10)
        ax.set_title(f"{ENV_LABELS.get(env, env)}: Expert vs. Simple Dataset Profile",
                     fontsize=10, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

        _save_figure(fig, f"radar_{env}_expert_vs_simple.png", output_dir)


# ---------------------------------------------------------------------------
# Plot 8: Radar Chart — All datasets in one plot
# ---------------------------------------------------------------------------


def plot_all_datasets_radar(df_profiles: pd.DataFrame, output_dir: str):
    """One radar chart per environment, with each tier (simple/medium/expert)
    shown as a separate coloured polygon."""
    metrics = _usable_features(df_profiles, COVERAGE_FEATURES)
    df = df_profiles.dropna(subset=metrics).copy()
    df = df[df["env_family"].notna() & df["tier"].notna()]
    if df.empty or not metrics:
        print("  [!] No profile data for per-environment radar charts.")
        return

    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]
    radar_labels = [_wrap_label(FEATURE_LABELS.get(m, m))
                    for m in metrics]

    # Tier-specific colours (sequential palette: simple→medium→expert)
    tier_colors = {
        # our own tiers
        "simple": "#fee5d9",  # light orange
        "medium": "#fcae91",  # medium orange
        "medium-replay": "#fb6a4a",  # darker orange
        "expert": "#de2d26",  # red
        # D4RL v0 tiers used by the d3rlpy paper source — mapped onto the
        # same weakest -> strongest colour scale as their counterparts above.
        "random": "#fee5d9",
        "medium-expert": "#de2d26",
    }
    tier_order = TIER_ORDER

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
                # Normalise using global min-max (across ALL datasets)
                values_norm = []
                for m in metrics:
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
        ax.set_xticklabels(radar_labels, fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=10)
        ax.set_title(f"{ENV_LABELS.get(env, env)}: Dataset Profile by Tier",
                     fontsize=10, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1),
                  fontsize=10, frameon=True)

        _save_figure(fig, f"radar_{env}_all_tiers.png", output_dir)


# ---------------------------------------------------------------------------
# Plot 9: Cross-Algorithm Tier Comparison — one figure per environment, one
# row (subplot) per dataset tier, each row a grouped bar chart of every
# algorithm's score plus a "Mean" bar
# ---------------------------------------------------------------------------


def plot_cross_algorithm_tier_comparison(
    df_scores: pd.DataFrame,
    output_dir: str,
    source_label: str,
):
    """For each environment, draw one figure with one row per dataset tier.
    Each row is a bar chart of every algorithm's D4RL score for that (env,
    tier), plus one extra "Mean" bar averaging across algorithms. Individual
    algorithm bars use their own seed std as the error bar; the Mean bar
    uses the std across algorithm means."""
    df = df_scores.dropna(subset=["mean_d4rl"]).copy()
    if df.empty:
        print("  [!] No valid performance data for cross-algorithm comparison.")
        return

    algorithms = sorted(df["algorithm"].unique())
    algo_colors = dict(
        zip(algorithms, sns.color_palette("muted", n_colors=len(algorithms))))
    mean_color = "#4d4d4d"

    for env in ENV_ORDER:
        env_df = df[df["env_family"] == env]
        if env_df.empty:
            continue

        tiers = [t for t in TIER_ORDER if t in env_df["tier"].unique()]
        if not tiers:
            continue

        fig, axes = plt.subplots(
            len(tiers), 1,
            figsize=(max(8.0, 1.1 * (len(algorithms) + 1)), 3.5 * len(tiers)),
            squeeze=False,
            sharey=True,  # same y-scale on every row so tiers are directly comparable
        )

        for ax, tier in zip(axes[:, 0], tiers):
            tier_df = env_df[env_df["tier"] == tier].set_index("algorithm")
            present_algos = [a for a in algorithms if a in tier_df.index]

            means = [tier_df.loc[a, "mean_d4rl"] for a in present_algos]
            stds = [tier_df.loc[a, "std_d4rl"] for a in present_algos]
            mean_of_means = float(np.mean(means))
            std_of_means = float(np.std(means))

            labels = present_algos + ["Mean"]
            values = means + [mean_of_means]
            errors = stds + [std_of_means]
            colors = [algo_colors[a] for a in present_algos] + [mean_color]

            x = np.arange(len(labels))
            bars = ax.bar(x, values, yerr=errors, capsize=3, color=colors,
                          error_kw={"linewidth": 1})
            bars[-1].set_edgecolor("black")
            bars[-1].set_linewidth(1.5)

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
            ax.set_ylabel("D4RL Score (%)", fontsize=10)
            ax.set_title(TIER_LABELS.get(tier, tier), fontsize=10, loc="left")
            ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle(
            f"{ENV_LABELS.get(env, env)}: Cross-Algorithm Performance by Tier ({source_label})",
            fontsize=10)
        fig.tight_layout()

        _save_figure(fig, f"{env}_tier_comparison.png", output_dir)


# ---------------------------------------------------------------------------
# Plot 10: Dataset Source Comparison — one figure per environment, one row
# per difficulty-rank tier group, each row a grouped bar chart of dataset
# meta-features comparing every dataset source (e.g. d4rl vs minari)
# ---------------------------------------------------------------------------


def _draw_grouped_source_bars(
    ax, group_df: pd.DataFrame, sources: list, source_colors: dict,
    features: list, x: np.ndarray, width: int, n_sources: int,
    error_feature: str = None, error_col: str = None,
):
    """Draw one panel of grouped bars: one bar per (source, feature), with
    an optional error bar on a single named feature (e.g. Mean_Return's
    error bar sourced from its companion Std_Return column). Returns the
    list of (source, BarContainer) pairs actually drawn, so callers that
    need per-bar value labels (e.g. the Size sub-panels) can bar_label()
    them."""
    containers = []
    for i, source in enumerate(sources):
        source_row = group_df[group_df["source"] == source]
        if source_row.empty:
            continue
        row = source_row.iloc[0]
        values = [row.get(f, np.nan) for f in features]
        errors = None
        if error_feature is not None and error_col is not None:
            errors = [row.get(error_col, np.nan) if f ==
                      error_feature else 0 for f in features]
        offset = (i - (n_sources - 1) / 2) * width
        container = ax.bar(x + offset, values, width, yerr=errors, capsize=3,
               label=source, color=source_colors[source], error_kw={"linewidth": 1})
        containers.append((source, container))
    return containers


def plot_dataset_source_comparison(df_profiles: pd.DataFrame, output_dir: str):
    """For each environment, draw one figure with one row per
    difficulty-rank tier group (TIER_GROUP_ORDER). Each row has a grouped
    bar-chart panel per dataset source per feature (e.g. d4rl vs minari) for
    Coverage features (COVERAGE_FEATURES) and Quality features
    (QUALITY_FEATURES), plus one independently-scaled sub-panel per Size
    feature (SIZE_FEATURES: Transitions, Episodes, Avg_Episode_Length).
    Coverage/Quality panels share a linear axis across their own features
    because those features are roughly comparable scale; Size features are
    related (Transitions ~= Episodes * Avg_Episode_Length) but differ from
    each other by orders of magnitude, so each gets its own linear axis
    (never log — these are exact counts, not measurements) with its bars
    labeled with the plain comma-grouped value, which both keeps every
    metric readable and preserves the exact figures a shared/log axis would
    obscure. Every Coverage/Quality row gets its own x-tick labels,
    word-wrapped at each space (not rotated) so multi-word feature names
    stay compact and legible as plain horizontal text; Size sub-panels have
    no x-axis (one bar-group per panel) and are headed by the metric name
    instead.

    Unlike the other profile plots, this one spans every dataset source at
    once, so a feature missing entirely for one source (e.g. EAS/ERI not yet
    computed for D4RL) shouldn't get excluded outright — that would hide it
    for every source, including ones that do have it. Rows are dropped only
    if ALL of COVERAGE_FEATURES are missing; a source simply gets no bar
    drawn for whichever individual feature it lacks (see
    _draw_grouped_source_bars, which already tolerates a NaN value for one
    feature)."""
    df = df_profiles.dropna(subset=COVERAGE_FEATURES, how="all").copy()
    if df.empty:
        print("  [!] No profile data for dataset-source comparison.")
        return
    df["tier_group"] = df["tier"].map(TIER_TO_GROUP)
    df = df.dropna(subset=["tier_group"])

    sources = sorted(df["source"].unique())
    source_colors = dict(
        zip(sources, sns.color_palette("muted", n_colors=len(sources))))
    coverage_labels = [_wrap_label(FEATURE_LABELS.get(f, f))
                       for f in COVERAGE_FEATURES]
    quality_labels = [_wrap_label(FEATURE_LABELS.get(f, f))
                      for f in QUALITY_FEATURES]
    size_headers = [_wrap_label(FEATURE_LABELS.get(f, f))
                    for f in SIZE_FEATURES]

    n_sources = len(sources)
    n_size = len(SIZE_FEATURES)
    width = 0.8 / n_sources
    x_coverage = np.arange(len(COVERAGE_FEATURES))
    x_quality = np.arange(len(QUALITY_FEATURES))
    x_single = np.arange(1)  # each Size sub-panel holds exactly one feature

    for env in ENV_ORDER:
        env_df = df[df["env_family"] == env]
        if env_df.empty:
            continue

        groups = [g for g in TIER_GROUP_ORDER if g in env_df["tier_group"].unique()]
        if not groups:
            continue

        # Column widths proportional to each panel's bar-category count, so
        # a single-feature panel doesn't get stretched to the same width as
        # a five-feature panel — every bar slot ends up roughly the same
        # width across panels. Each Size sub-panel counts as a 1-feature
        # column, same as a single-feature Coverage/Quality panel would.
        panel_feature_counts = [len(COVERAGE_FEATURES),
                                 len(QUALITY_FEATURES)] + [1] * n_size
        n_cols = 2 + n_size
        fig, axes = plt.subplots(
            len(groups), n_cols,
            figsize=(max(13.0, 1.6 * sum(panel_feature_counts)),
                     3.0 * len(groups)),
            squeeze=False,
            gridspec_kw={"width_ratios": panel_feature_counts},
            # Share the y-axis within each panel column across every
            # tier-group row, so e.g. the "Simple / Random" row's Coverage
            # panel is scaled identically to the "Medium" row's Coverage
            # panel and bar heights are directly comparable top-to-bottom.
            # Columns still scale independently of each other since
            # Coverage/Quality/each Size metric live on very different
            # magnitudes.
            sharey="col",
        )

        for row_idx, group in enumerate(groups):
            group_df = env_df[env_df["tier_group"] == group]
            ax_coverage, ax_quality = axes[row_idx, 0], axes[row_idx, 1]
            size_axes = axes[row_idx, 2:2 + n_size]

            _draw_grouped_source_bars(
                ax_coverage, group_df, sources, source_colors,
                COVERAGE_FEATURES, x_coverage, width, n_sources)
            _draw_grouped_source_bars(
                ax_quality, group_df, sources, source_colors,
                QUALITY_FEATURES, x_quality, width, n_sources)

            for ax, x, labels, col_label in (
                (ax_coverage, x_coverage, coverage_labels, "Coverage"),
                (ax_quality, x_quality, quality_labels, "Quality"),
            ):
                ax.set_title(TIER_GROUP_LABELS.get(
                    group, group), fontsize=10, loc="left")
                # Column category header, set once on the top row only. Uses
                # a plain axes-fraction text rather than a second set_title()
                # call: matplotlib shares one title-offset transform across
                # the 'left'/'center'/'right' title slots of an axes, so a
                # second set_title() call's pad overrides the first and both
                # end up on the same baseline instead of stacking.
                if row_idx == 0:
                    ax.text(0.5, 1.22, col_label, transform=ax.transAxes,
                            ha="center", va="bottom", fontsize=10,
                            fontweight="bold")
                ax.grid(axis="y", alpha=0.3)
                ax.set_xticks(x)
                ax.set_xticklabels(
                    labels, rotation=0, ha="center", fontsize=10)

            for feature, header, ax in zip(SIZE_FEATURES, size_headers, size_axes):
                containers = _draw_grouped_source_bars(
                    ax, group_df, sources, source_colors,
                    [feature], x_single, width, n_sources)
                for _, container in containers:
                    ax.bar_label(
                        container,
                        fmt=lambda v, feat=feature: _format_size_value(v, feat),
                        fontsize=7, padding=2)
                # Plain comma-grouped y-tick labels rather than matplotlib's
                # default scientific/offset notation for large counts — see
                # _format_size_value.
                ax.yaxis.set_major_formatter(
                    FuncFormatter(lambda v, _, feat=feature: _format_size_value(v, feat)))
                ax.set_title(TIER_GROUP_LABELS.get(
                    group, group), fontsize=10, loc="left")
                if row_idx == 0:
                    ax.text(0.5, 1.22, header, transform=ax.transAxes,
                            ha="center", va="bottom", fontsize=10,
                            fontweight="bold")
                ax.grid(axis="y", alpha=0.3)
                ax.set_xticks([])  # one bar-group per panel; header names it

        # Headroom above each Size column's tallest bar so its value label
        # isn't clipped. Set once per column, after every row has been
        # drawn, on the row-0 axis: sharey="col" links view limits across
        # the column, so this widens the shared range for every row.
        for j in range(n_size):
            ax0 = axes[0, 2 + j]
            top = ax0.get_ylim()[1]
            ax0.set_ylim(bottom=0, top=top * 1.18)

        # Built directly from source_colors (rather than pulled off one
        # subplot's handles) so every source appears even if a row happens
        # to be missing one of them (e.g. a d4rl-only "medium-replay" row).
        legend_handles = [plt.Rectangle((0, 0), 1, 1, color=source_colors[s])
                          for s in sources]
        fig.legend(legend_handles, sources, loc="upper right", fontsize=10)

        fig.suptitle(
            f"{ENV_LABELS.get(env, env)}: Dataset Source Comparison by Tier",
            fontsize=10)
        fig.tight_layout()

        _save_figure(fig, f"{env}_source_comparison.png", output_dir)


# ---------------------------------------------------------------------------
# Plot orchestration — selection (which source) stays separate from
# rendering (the plot_* functions above, which never branch on source).
# ---------------------------------------------------------------------------


def generate_score_dependent_figures(
    df_results: pd.DataFrame,
    df_meta: pd.DataFrame,
    output_dir: str,
    algorithm_label: str,
):
    """Generate every figure whose content depends on a specific (result
    source, algorithm) pair, saved into output_dir (already namespaced by the
    caller as .../result_dependent/RESULT_SOURCE/ALGORITHM/) so different
    sources/algorithms never overwrite each other's figures."""
    print(f"\n  --- {algorithm_label}  ->  {output_dir} ---")

    print("  • Performance Overview (grouped bar chart)...")
    plot_performance_overview(df_results, output_dir,
                              algorithm_label=algorithm_label)

    print("  • Metrics vs. Performance (scatter + trend)...")
    plot_metrics_vs_performance(
        df_meta, output_dir, algorithm_label=algorithm_label)

    print("  • Correlation Heatmap...")
    plot_correlation_heatmap(
        df_meta, output_dir, algorithm_label=algorithm_label)

    print("  • Feature Importance (Random Forest)...")
    plot_feature_importance(df_meta, output_dir,
                            algorithm_label=algorithm_label)

    print("  • Predicted vs. Actual (LOO-CV)...")
    plot_predicted_vs_actual(
        df_meta, output_dir, algorithm_label=algorithm_label)

    print("  • Seed Consistency (boxplot)...")
    plot_seed_consistency(df_results, output_dir)


def generate_source_independent_figures(df_profiles: pd.DataFrame, output_dir: str):
    """Generate figures that depend only on dataset profiles, never on any
    algorithm's performance, so they're identical across every result
    source. Saved into output_dir (already namespaced by the caller as
    .../result_independent/DATASET_SOURCE/)."""
    print("\n  • Radar: Expert vs. Simple per Environment...")
    plot_radar_comparison(df_profiles, output_dir)

    print("\n  • Radar: All Tiers per Environment...")
    plot_all_datasets_radar(df_profiles, output_dir)


def generate_cross_algorithm_figures(df_all_scores: pd.DataFrame, output_dir: str, source_label: str):
    """Generate figures that compare every algorithm within a single result
    source against each other, saved into output_dir (already namespaced by
    the caller as .../cross_algorithm/RESULT_SOURCE/)."""
    print("\n  • Cross-Algorithm Tier Comparison (grouped bars per tier)...")
    plot_cross_algorithm_tier_comparison(
        df_all_scores, output_dir, source_label=source_label)


def generate_dataset_source_comparison_figures(df_all_profiles: pd.DataFrame, output_dir: str):
    """Generate figures that compare every dataset source (e.g. d4rl vs
    minari) against each other, saved into output_dir (already namespaced by
    the caller as .../dataset_source_comparison/)."""
    print("\n  • Dataset Source Comparison (grouped bars per tier)...")
    plot_dataset_source_comparison(df_all_profiles, output_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():

    # Resolve paths relative to the project root — default_paths.py constants
    # are expressed relative to it (e.g. "src\\results\\...").
    policy_report_dir = os.path.normpath(
        os.path.join(_PROJECT_ROOT, default_paths.POLICY_RESULTS_DIR))
    profiles_dir = os.path.normpath(
        os.path.join(_PROJECT_ROOT, default_paths.DATASET_PROFILES_DIR))
    output_dir = os.path.normpath(os.path.join(
        _PROJECT_ROOT, default_paths.DEFAULT_FIGURE_OUTPUT_DIR))

    # ---- Load data shared across every source ----
    print("\n[1/2] Loading dataset profiles & policy result reports...")

    # the structure is profiles_dir/result_independent/DATASET_SOURCE/DATASET/TIER/...
    dataset_profile_data: dict = _load_all_dataset_profiles(profiles_dir)
    # the structure is policy_report_dir/result_dependent/RESULT_SOURCE/ALGORITHM/DATASET/TIER/...
    policy_score_data: dict = _load_all_policy_reports(policy_report_dir)

    print(
        f"  Dataset profile sources found: {sorted(dataset_profile_data.keys())}")
    print(
        f"  Policy result sources found:   {sorted(policy_score_data.keys())}")

    print("\n[2/2] Generating figures...")

    # the structure of the result independent plots is output_dir/result_independent/DATASET_SOURCE/ => for result independent plots
    # the structure of the result dependent plots is output_dir/result_dependent/RESULT_SOURCE/ALGORITHM/ => for result dependent plots
    # only create plots for dataset sources that have both profiles and policy reports
    for profile_source, profiles in dataset_profile_data.items():
        matching_result_sources = [
            result_source
            for result_source, mapped_profile_source in RESULT_SOURCE_TO_PROFILE_SOURCE.items()
            if mapped_profile_source == profile_source and policy_score_data.get(result_source)
        ]

        if not profiles or not matching_result_sources:
            print(
                f"\n[!] Skipping dataset source '{profile_source}': no profiles "
                f"or no matching policy result source with data.")
            continue

        df_profiles = _profiles_dict_to_df(profiles)

        print(f"\n--- Dataset source '{profile_source}' ---")
        generate_source_independent_figures(
            df_profiles, os.path.join(output_dir, "result_independent", profile_source))

        for result_source in matching_result_sources:
            for algorithm in policy_score_data[result_source]:
                df_results = _scores_dict_to_df(
                    policy_score_data[result_source], algorithm)
                if df_results.dropna(subset=["mean_d4rl"]).empty:
                    print(
                        f"\n[!] Skipping '{result_source}/{algorithm}': no valid scores.")
                    continue

                df_meta = _build_meta_df(df_profiles, df_results)
                algo_output_dir = os.path.join(
                    output_dir, "result_dependent", result_source, _slugify_algorithm(algorithm))
                generate_score_dependent_figures(
                    df_results, df_meta, algo_output_dir, algorithm_label=algorithm)

    # the structure of the cross-algorithm plots is output_dir/cross_algorithm/RESULT_SOURCE/
    # these only need policy scores (no dataset profiles), grouped per result
    # source so algorithms sharing a tier vocabulary get compared together
    # (e.g. every d3rlpy_paper baseline, but not against our own self_trained runs)
    for result_source, algorithms in policy_score_data.items():
        df_all_scores = _all_scores_dict_to_df(algorithms)
        if df_all_scores.dropna(subset=["mean_d4rl"]).empty:
            continue

        print(f"\n--- Cross-algorithm comparison: '{result_source}' ---")
        generate_cross_algorithm_figures(
            df_all_scores,
            os.path.join(output_dir, "cross_algorithm", result_source),
            source_label=result_source,
        )

    # the structure of the dataset-source comparison plots is output_dir/dataset_source_comparison/
    # these only need dataset profiles (no policy scores), compared across
    # every dataset source at once (e.g. d4rl vs minari)
    df_all_profiles = _all_profiles_dict_to_df(dataset_profile_data)
    if not df_all_profiles.dropna(subset=COVERAGE_FEATURES).empty:
        print("\n--- Dataset source comparison ---")
        generate_dataset_source_comparison_figures(
            df_all_profiles, os.path.join(output_dir, "dataset_source_comparison"))

    print("\n[✓] Done.")


if __name__ == "__main__":
    main()
