import os
import json
import numpy as np
import d3rlpy

from configs import default_paths

from dataset_analyzer.profile_feature_computer import ProfileFeatureComputer


PRINT_VERBOSE = True
PLOT_HISTOGRAMS = False

CALCULATE_EAS = False


def load_d4rl_dataset(env_name):

    # Natively load d4rl datasets via d3rlpy ecosystem
    dataset, _ = d3rlpy.datasets.get_d4rl(env_name)

    return [
        {
            "observations": np.array(episode.observations),
            "actions": np.array(episode.actions),
            "rewards": np.array(episode.rewards)
        }
        for episode in dataset.episodes
    ]


def load_minari_dataset(env_name):

    # Natively load d4rl datasets via d3rlpy ecosystem
    dataset, _ = d3rlpy.datasets.get_minari(env_name)

    return [
        {
            "observations": np.array(episode.observations),
            "actions": np.array(episode.actions),
            "rewards": np.array(episode.rewards)
        }
        for episode in dataset.episodes
    ]


def save_profile(profile, output_dir=default_paths.DATASET_PROFILES_DIR):
    os.makedirs(output_dir, exist_ok=True)

    # Clean the slash characters from d4rl path strings for valid file systems
    name = profile["Dataset"].replace("/", "_")
    path = os.path.join(output_dir, f"{name}.json")

    with open(path, "w") as file:
        json.dump(profile, file, indent=4)

    print(f"[+] Saved: {path}")


def plot_feature_histograms(data, feature_label, save_path, n_bins=50):
    """Plot one histogram per feature (column) of `data`, each as its own
    subplot within a single matplotlib figure, and save it to `save_path`."""
    import matplotlib.pyplot as plt

    n_features = data.shape[1]
    n_cols = min(4, n_features)
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)

    for feature in range(n_features):
        row, col = divmod(feature, n_cols)
        ax = axes[row][col]
        ax.hist(data[:, feature], bins=n_bins,
                color="#2a78d6", edgecolor="#fcfcfb", linewidth=0.5)
        ax.set_title(f"{feature_label} {feature}",
                     fontsize=10, color="#0b0b0b")
        ax.set_xlabel("Value", fontsize=8, color="#52514e")
        ax.set_ylabel("Count", fontsize=8, color="#52514e")
        ax.tick_params(labelsize=7, colors="#898781")
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.5)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#c3c2b7")

    # Hide any unused subplot slots when n_features doesn't fill the grid
    for empty in range(n_features, n_rows * n_cols):
        row, col = divmod(empty, n_cols)
        axes[row][col].axis("off")

    fig.patch.set_facecolor("#fcfcfb")
    fig.suptitle(f"{feature_label} Distributions ({n_bins} bins)",
                 fontsize=12, color="#0b0b0b")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"[+] Saved: {save_path}")


def plot_histograms(states, actions, env_name, hist_bins=50, output_dir=default_paths.DATASET_PROFILES_DIR):
    # create histogram of state and action distributions and plot them
    hist_dir = os.path.join(output_dir, "histograms")
    os.makedirs(hist_dir, exist_ok=True)
    name = env_name.replace("/", "_")

    plot_feature_histograms(
        states, "State Feature",
        os.path.join(
            hist_dir, f"state_features\\{name}_state_histograms.png"),
        n_bins=hist_bins
    )
    plot_feature_histograms(
        actions, "Action Feature",
        os.path.join(
            hist_dir, f"action_features\\{name}_action_histograms.png"),
        n_bins=hist_bins
    )


def load_episodes(env_name):
    if "mujoco" in env_name.lower() or "minari" in env_name.lower():
        return load_minari_dataset(env_name)
    return load_d4rl_dataset(env_name)


def dataset_family(env_name):
    """Group sibling dataset variants of the same environment (e.g. the quality/size
    variants of hopper) so SACo discretization bounds can be pooled across them."""
    if "/" in env_name:
        # e.g. "mujoco/hopper/medium-v0" -> "mujoco/hopper"
        return env_name.rsplit("/", 1)[0]
    # e.g. "hopper-medium-v0" -> "hopper"
    return env_name.split("-", 1)[0]


def compute_family_saco_bounds(env_names):
    """Pool state/action bounds across all sibling dataset variants so SACo
    discretization uses a shared range instead of each dataset's own min/max."""
    all_states, all_actions = [], []
    for env_name in env_names:
        episodes = load_episodes(env_name)
        all_states.append(np.concatenate(
            [episode["observations"] for episode in episodes]))
        actions = np.concatenate(
            [episode["actions"] for episode in episodes])
        if actions.ndim == 1:
            actions = actions.reshape(-1, 1)
        all_actions.append(actions)

    return ProfileFeatureComputer.compute_state_action_bounds(
        np.concatenate(all_states), np.concatenate(all_actions))


def analyze_dataset(env_name, n_clusters=20, hist_bins=50, saco_bins=10, saco_bounds=None, output_dir=default_paths.DATASET_PROFILES_DIR):
    print(f"\n[+] Loading {env_name}")

    episodes = load_episodes(env_name)

    states = np.concatenate(
        [episode["observations"] for episode in episodes]
    )
    actions = np.concatenate(
        [episode["actions"] for episode in episodes]
    )
    # Ensure actions are always 2D (N, D) so downstream indexing is safe
    # for discrete / 1-D action spaces too.
    if actions.ndim == 1:
        actions = actions.reshape(-1, 1)

    rewards = np.concatenate(
        [episode["rewards"] for episode in episodes]
    )
    returns = np.array(
        [np.sum(ep["rewards"]) for ep in episodes]
    )
    lengths = np.array(
        [len(ep["rewards"]) for ep in episodes]
    )

    if PRINT_VERBOSE:
        print(f"[+] States shape: {states.shape}")
        print(f"[+] Actions shape: {actions.shape}")
        print(f"[+] Rewards shape: {rewards.shape}")
        print(f"[+] Returns shape: {returns.shape}")
        print(f"[+] Lengths shape: {lengths.shape}")

    if PRINT_VERBOSE:
        for state_feature in range(states.shape[1]):
            print(f"[+] State Feature {state_feature} | min = {np.min(states[:, state_feature]):.4f}, max = {np.max(states[:, state_feature]):.4f}, mean = {np.mean(states[:, state_feature]):.4f}, std = {np.std(states[:, state_feature]):.4f}")

        for action_feature in range(actions.shape[1]):
            print(f"[+] Action Feature {action_feature} | min = {np.min(actions[:, action_feature]):.4f}, max = {np.max(actions[:, action_feature]):.4f}, mean = {np.mean(actions[:, action_feature]):.4f}, std = {np.std(actions[:, action_feature]):.4f}")

    if PRINT_VERBOSE:
        print(
            f"[+] Reward | min = {np.min(rewards):.4f}, max = {np.max(rewards):.4f}, mean = {np.mean(rewards):.4f}, std = {np.std(rewards):.4f}")

    if PLOT_HISTOGRAMS:
        plot_histograms(states, actions, env_name,
                        hist_bins=hist_bins, output_dir=output_dir)

    n_state_clusters = min(n_clusters, len(states))
    n_traj_clusters = min(n_clusters, len(episodes))

    min_return = float(np.min(returns))
    max_return = float(np.max(returns))
    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns))
    med_return = float(np.median(returns))
    reward_sparcity = float(np.mean(rewards == 0))

    # State Coverage
    state_spread, state_cluster_coverage, state_entropy_coverage, action_variance, action_entropy = ProfileFeatureComputer.compute_state_coverage(
        states, actions, n_state_clusters)

    # mean Estimated Action Stochasticity EAS and normalized Expected Return Index ERI as in Swazinna et al
    if CALCULATE_EAS:
        eas = ProfileFeatureComputer.compute_mean_estimated_action_stochasticity(
            actions, states)
    else:
        eas = 0.0

    eri = ProfileFeatureComputer.compute_normalized_ERI(
        min_return,
        max_return,
        mean_return
    )

    # trajectory diversity
    trajectory_diversity = ProfileFeatureComputer.compute_trajectory_diversity(
        episodes, n_traj_clusters)

    # state action coverage inspired by Schweighofer et al, but without replay normalisation but with continous state and action space quantization
    saco = ProfileFeatureComputer.compute_relative_state_action_coverage(
        states, actions, saco_bins, bounds=saco_bounds)

    # relative trajectory quality as proposed by Schweighofer et al
    tq = ProfileFeatureComputer.compute_TQ(min_return, max_return, mean_return)

    return {
        "Dataset": env_name,

        "Size": {
            "Transitions": int(len(actions)),
            "Episodes": int(len(episodes)),
            "Average Episode Length": float(np.mean(lengths))
        },

        "Coverage": {
            "State Spread": state_spread,
            "State Cluster Coverage": state_cluster_coverage,
            "State Entropy": state_entropy_coverage,
            "Action Variance": action_variance,
            "Action Entropy": action_entropy,
            "EAS": eas,  # Swizanna et al
            "SACo": saco  # ~ Schweighofer et al
        },

        "Quality": {
            "Mean Return": mean_return,
            "Std Return": std_return,
            "Min Return": min_return,
            "Max Return": max_return,
            "Median Return": med_return,
            "Reward Sparsity": reward_sparcity,
            "ERI": eri,  # Swizanna et al
            "TQ": tq  # Schweighofer et al
        },

        "Diversity": {
            "Trajectory Diversity": trajectory_diversity
        }
    }


if __name__ == "__main__":

    # minari profiles
    minari_datasets = [
        # --- 1. Walker2d Suite ---
        "mujoco/walker2d/simple-v0",
        "mujoco/walker2d/medium-v0",
        "mujoco/walker2d/expert-v0",

        # --- 2. HalfCheetah Suite ---
        "mujoco/halfcheetah/simple-v0",
        "mujoco/halfcheetah/medium-v0",
        "mujoco/halfcheetah/expert-v0",

        # --- 3. Hopper Suite ---
        "mujoco/hopper/simple-v0",
        "mujoco/hopper/medium-v0",
        "mujoco/hopper/expert-v0",
        "mujoco/hopper/medium-replay-v0",

        # --- 4. Ant Suite ---
        "mujoco/ant/simple-v0",
        "mujoco/ant/medium-v0",
        "mujoco/ant/expert-v0",

        # # --- 5. Humanoid Suite ---
        # "mujoco/humanoid/simple-v0",
        # "mujoco/humanoid/medium-v0",
        # "mujoco/humanoid/expert-v0"
    ]
    # d4rl profiles
    d4rl_datasets = [
        # Antmaze
        'antmaze-large-play-v0',
        'antmaze-medium-play-v0',
        'antmaze-umaze-v0',
        # Halfcheetah
        'halfcheetah-medium-expert-v0',
        'halfcheetah-medium-replay-v0',
        'halfcheetah-medium-v0',
        'halfcheetah-random-v0',
        # Hopper
        'hopper-medium-expert-v0',
        'hopper-medium-replay-v0',
        'hopper-medium-v0',
        'hopper-random-v0',
        # Walker2d
        'walker2d-medium-expert-v0',
        'walker2d-medium-replay-v0',
        'walker2d-medium-v0',
        'walker2d-random-v0',
    ]

    # Each dataset source gets its own subdirectory under
    # DATASET_PROFILES_DIR (profiles + histograms), matching the existing
    # src/results/dataset_profiles/{d4rl,minari}/ layout.
    dataset_sources = [
        (minari_datasets, os.path.join(
            default_paths.DATASET_PROFILES_DIR, "minari")),
        (d4rl_datasets, os.path.join(
            default_paths.DATASET_PROFILES_DIR, "d4rl")),
    ]

    for datasets, output_dir in dataset_sources:
        families = {}
        for dataset in datasets:
            families.setdefault(dataset_family(dataset), []).append(dataset)

        for family, members in families.items():
            try:
                print(
                    f"[+] Pooling SACo bounds for '{family}' ({len(members)} variants)")
                saco_bounds = compute_family_saco_bounds(members)
            except Exception as e:
                print(
                    f"Could not pool SACo bounds for '{family}', falling back to per-dataset bounds; {e}")
                saco_bounds = None

            for dataset in members:
                try:
                    print(f"[+] Analyzing {dataset}")
                    profile = analyze_dataset(
                        dataset, output_dir=output_dir, saco_bounds=saco_bounds)
                    # break
                    print(json.dumps(profile, indent=4))
                    save_profile(profile, output_dir=output_dir)
                except Exception as e:
                    print(
                        f"Could not analyse dataset {dataset}. The pipeline yielded an exception; {e}")
