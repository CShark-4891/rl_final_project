import os
import json
import numpy as np
import d3rlpy

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy


def save_profile(profile, output_dir="dataset_profiles"):
    os.makedirs(output_dir, exist_ok=True)

    # Clean the slash characters from Minari path strings for valid file systems
    name = profile["Dataset"].replace("/", "_")
    path = os.path.join(output_dir, f"{name}.json")

    with open(path, "w") as file:
        json.dump(profile, file, indent=4)

    print(f"[+] Saved: {path}")


def calculate_entropy(values, bins=50, value_range=None):
    histogram, _ = np.histogram(
        values, bins=bins, range=value_range, density=True)
    histogram = histogram[histogram > 0]

    if len(histogram) == 0:
        return 0.0

    return float(entropy(histogram))


def normalized_cluster_entropy(labels, n_clusters):
    """Entropy of the cluster occupancy distribution, normalized to [0, 1]."""
    if n_clusters <= 1:
        return 0.0

    counts = np.bincount(labels, minlength=n_clusters)
    probs = counts / len(labels)
    probs = probs[probs > 0]

    return float(entropy(probs) / np.log(n_clusters))


def load_minari_dataset(env_name):

    # Natively load Minari datasets via d3rlpy ecosystem
    dataset, _ = d3rlpy.datasets.get_minari(env_name)

    return [
        {
            "observations": np.array(episode.observations),
            "actions": np.array(episode.actions),
            "rewards": np.array(episode.rewards)
        }
        for episode in dataset.episodes
    ]


def analyze_dataset(env_name, clusters=20):
    print(f"\n[+] Loading {env_name}")

    episodes = load_minari_dataset(env_name)

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
        "Reward Sparsity": float(np.mean(rewards == 0))
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
    # Updated target targets matching standard modern Minari profiles
    datasets = [
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

        # --- 5. Humanoid Suite ---
        "mujoco/humanoid/simple-v0",
        "mujoco/humanoid/medium-v0",
        "mujoco/humanoid/expert-v0"
    ]

    for dataset in datasets:
        print(f"[+] Analyzing {dataset}")
        profile = analyze_dataset(dataset)
        print(json.dumps(profile, indent=4))
        save_profile(profile)
