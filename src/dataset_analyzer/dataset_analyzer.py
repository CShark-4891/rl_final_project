import os
import json
import numpy as np
import d3rlpy

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy

from configs import default_paths

PRINT_VERBOSE = True


def save_profile(profile, output_dir=default_paths.DATASET_PROFILES_DIR):
    os.makedirs(output_dir, exist_ok=True)

    # Clean the slash characters from d4rl path strings for valid file systems
    name = profile["Dataset"].replace("/", "_")
    path = os.path.join(output_dir, f"{name}.json")

    with open(path, "w") as file:
        json.dump(profile, file, indent=4)

    print(f"[+] Saved: {path}")


def calculate_entropy(values, bins=50, value_range=None):
    histogram, _ = np.histogram(
        values, bins=bins, range=value_range, density=False)
    histogram = histogram / histogram.sum()  # Normalize to probabilities

    if len(histogram) == 0 or histogram.sum() == 0:
        return 0.0

    histogram = histogram[histogram > 0]
    return float(entropy(histogram))


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


def normalized_cluster_entropy(labels, n_clusters):
    """Entropy of the cluster occupancy distribution, normalized to [0, 1]."""
    if n_clusters <= 1:
        return 0.0

    counts = np.bincount(labels, minlength=n_clusters)
    probs = counts / len(labels)
    probs = probs[probs > 0]

    return float(entropy(probs) / np.log(n_clusters))


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


def analyze_dataset(env_name, clusters=20, hist_bins=50, output_dir=default_paths.DATASET_PROFILES_DIR):
    print(f"\n[+] Loading {env_name}")

    episodes = load_d4rl_dataset(env_name)

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

    n_state_clusters = min(clusters, len(states))
    n_traj_clusters = min(clusters, len(episodes))

    # Metric 1.1: State coverage
    state_spread = float(np.mean(np.std(states, axis=0)))

    # Scale states before clustering so high-variance dimensions don't
    # dominate the cluster assignment
    state_scaler = StandardScaler()
    scaled_states = state_scaler.fit_transform(states)

    state_model = MiniBatchKMeans(
        n_clusters=n_state_clusters,
        random_state=42,
        n_init="auto"
    )
    state_labels = state_model.fit_predict(scaled_states)

    state_cluster_coverage = float(
        len(np.unique(state_labels)) / n_state_clusters)
    state_entropy_coverage = normalized_cluster_entropy(
        state_labels, n_state_clusters)

    # Metric 1.2: Action variance and entropy -> Action Coverage
    action_variance = float(np.mean(np.var(actions, axis=0)))
    action_entropy = float(np.mean([
        calculate_entropy(actions[:, i]) for i in range(actions.shape[1])
    ]))

    # Metric 2: Dataset quality
    quality = {
        "Mean Return": float(np.mean(returns)),
        "Std Return": float(np.std(returns)),
        "Min Return": float(np.min(returns)),
        "Max Return": float(np.max(returns)),
        "Median Return": float(np.median(returns)),
        # Reward sparsity: fraction of steps with near-zero reward
        # Use a threshold since continuous Mujoco rewards are rarely exactly 0
        "Reward Sparsity": float(np.mean(np.abs(rewards) < 0.1))
    }

    # Metric 3: Trajectory diversity
    trajectory_features = []

    for episode in episodes:
        ep_actions = episode["actions"]
        if ep_actions.ndim == 1:
            ep_actions = ep_actions.reshape(-1, 1)

        trajectory_features.append(np.concatenate(
            [np.mean(episode["observations"], axis=0),
             np.std(episode["observations"], axis=0),
             np.mean(ep_actions, axis=0),
             np.std(ep_actions, axis=0),
             [np.sum(episode["rewards"]), len(episode["rewards"])]]
        ))

    trajectory_features = np.array(trajectory_features)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(trajectory_features)

    # Run K-Means on scaled features
    trajectory_model = MiniBatchKMeans(
        n_clusters=n_traj_clusters,
        random_state=42,
        n_init="auto"
    )
    traj_labels = trajectory_model.fit_predict(scaled_features)

    trajectory_diversity = normalized_cluster_entropy(
        traj_labels, n_traj_clusters)

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
            "Action Entropy": action_entropy
        },

        "Quality": quality,

        "Diversity": {
            "Trajectory Diversity": trajectory_diversity
        }
    }


if __name__ == "__main__":
    # Updated target targets matching standard modern d4rl profiles
    d3rlpy_datasets = [
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

    for dataset in d3rlpy_datasets:
        print(f"[+] Analyzing {dataset}")
        profile = analyze_dataset(dataset)
        # break
        print(json.dumps(profile, indent=4))
        save_profile(profile)
