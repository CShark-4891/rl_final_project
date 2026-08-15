import numpy as np
import torch

from sklearn.cluster import MiniBatchKMeans
from scipy.stats import entropy

PRINT_VERBOSE = True


class ProfileFeatureComputer:

    def _calculate_entropy(values, bins=50, value_range=None):
        histogram, _ = np.histogram(
            values, bins=bins, range=value_range, density=True)
        histogram = histogram[histogram > 0]

        if len(histogram) == 0:
            return 0.0

        return float(entropy(histogram))

    def _normalized_cluster_entropy(labels, n_clusters):
        """Entropy of the cluster occupancy distribution, normalized to [0, 1]."""
        if n_clusters <= 1:
            return 0.0

        counts = np.bincount(labels, minlength=n_clusters)
        probs = counts / len(labels)
        probs = probs[probs > 0]

        return float(entropy(probs) / np.log(n_clusters))

    def compute_TQ(min_return, max_return, mean_return) -> float:
        """Relative Trajectory Quality (TQ) as in Understanding the Effects of Dataset Characteristics on Ofﬂine Reinforcement Learning by Schweighofer et al"""
        if max_return == min_return:
            return 0.0

        return float((mean_return - min_return) / (max_return - min_return))

    def compute_state_action_bounds(states, actions):
        """Elementwise (min, max) per dimension across a batch of states and actions.

        Intended to be pooled across all dataset variants of the same environment
        (e.g. every hopper-*-v0 dataset) and passed as `bounds` to
        compute_relative_state_action_coverage, so the discretization range is
        shared across variants rather than each dataset rescaling to its own
        min/max.
        """
        combined = np.concatenate(
            [np.asarray(states), np.asarray(actions)], axis=1)

        return combined.min(axis=0), combined.max(axis=0)

    def compute_relative_state_action_coverage(states, actions, n_bins=10, bounds=None) -> float:
        """Approximate State-Action Coverage (SACo) as in "Understanding the Effects of
        Dataset Characteristics on Offline Reinforcement Learning" by Schweighofer et al.

        The original SACo counts exact duplicate (s, a) pairs (via HyperLogLog for
        efficiency), which only produces meaningful duplicates for discrete or
        near-deterministic environments. For continuous state-action spaces almost
        every raw pair is unique, so each dimension is first independently
        discretized into `n_bins` equal-width bins; a "unique" pair is then a
        unique combination of per-dimension bin indices. SACo is reported as the
        fraction of transitions occupying a distinct cell (unique cells / N),
        i.e. one minus the duplication rate.

        `bounds`, if given, is a (min, max) pair of per-dimension arrays (see
        compute_state_action_bounds) that fixes the discretization range. This
        should be pooled across all dataset variants of the same environment:
        without a shared range, each dataset's bins stretch to fill only its own
        min/max, so a narrow dataset (e.g. expert) looks just as "covering" as a
        wide one (e.g. random) even though it explores far less of the space. If
        omitted, this dataset's own min/max is used instead, which is only
        meaningful when scoring a single dataset in isolation.
        """
        states = np.asarray(states)
        actions = np.asarray(actions)

        n_transitions = states.shape[0]
        if n_transitions == 0:
            return 0.0

        combined = np.concatenate([states, actions], axis=1)

        if bounds is None:
            min_vals, max_vals = combined.min(axis=0), combined.max(axis=0)
        else:
            min_vals, max_vals = bounds

        bin_indices = np.zeros_like(combined, dtype=np.int64)
        for dim in range(combined.shape[1]):
            min_val, max_val = min_vals[dim], max_vals[dim]
            if min_val == max_val:
                continue  # constant dimension: every sample stays in bin 0

            interior_edges = np.linspace(min_val, max_val, n_bins + 1)[1:-1]
            bin_indices[:, dim] = np.clip(
                np.digitize(combined[:, dim], interior_edges), 0, n_bins - 1)

        n_unique_cells = len(np.unique(bin_indices, axis=0))

        return float(n_unique_cells / n_transitions)

    def compute_normalized_ERI(min_return, max_return, mean_return) -> float:
        """Compute the normalized Expected Return Index (ERI) for a dataset as in "Measuring Data Quality for Data Selection in Offline Reinforcement Learning" by Swazinna et al (formula 2)."""
        if max_return == min_return:
            return 0.0  # Avoid division by zero; all returns are the same

        if min_return < 0:
            return float(((max_return - min_return) - (mean_return - min_return)) / (mean_return - min_return))
        else:
            return float(((max_return) - (mean_return)) / (mean_return))

    def compute_mean_estimated_action_stochasticity(actions, states, epochs=50, batch_size=256) -> float:
        """Estimate the Action Stochasticity (EAS) of the behavior policy that
        generated the dataset, as in "Measuring Data Quality for Data Selection in
        Offline Reinforcement Learning" by Swazinna et al (Eqs. 3-5). """

        class ASModel(torch.nn.Module):
            def __init__(self, state_dim, action_dim):
                super().__init__()
                # Shared feature map phi(s): two hidden layers of size 100, ReLU
                self.phi = torch.nn.Sequential(
                    torch.nn.Linear(state_dim, 100),
                    torch.nn.ReLU(),
                    torch.nn.Linear(100, 100),
                    torch.nn.ReLU(),
                )
                self.f_mu = torch.nn.Linear(100, action_dim)
                self.f_sigma = torch.nn.Linear(100, action_dim)

            def forward(self, s):
                features = self.phi(s)
                mu = self.f_mu(features)
                # Softplus keeps the predicted std strictly positive and smooth
                sigma = torch.nn.functional.softplus(
                    self.f_sigma(features)) + 1e-6
                return mu, sigma

        def train_AS_model(model, states, actions, epochs, batch_size, device):
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            eps = 1e-6

            dataset = torch.utils.data.TensorDataset(states, actions)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True)

            model.train()
            for epoch in range(epochs):
                epoch_loss, n_batches = 0.0, 0
                for batch_states, batch_actions in dataloader:
                    batch_states = batch_states.to(device)
                    batch_actions = batch_actions.to(device)

                    optimizer.zero_grad()
                    mu, sigma = model(batch_states)

                    # Invert the tanh squash from Eq. 3 to recover the pre-tanh
                    # action and add the resulting change-of-variables term, i.e.
                    # the standard tanh-Normal log-likelihood
                    squashed_actions = torch.clamp(
                        batch_actions, -1.0 + eps, 1.0 - eps)
                    pre_tanh_actions = torch.atanh(squashed_actions)

                    log_prob = torch.distributions.Normal(
                        mu, sigma).log_prob(pre_tanh_actions)
                    log_prob -= torch.log(1.0 - squashed_actions.pow(2) + eps)
                    log_prob = log_prob.sum(dim=-1)

                    # Eq. 4: minimize the negative log likelihood
                    loss = -log_prob.mean()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                if PRINT_VERBOSE:
                    print(
                        f"[+] AS model epoch {epoch + 1}/{epochs} | NLL = {epoch_loss / n_batches:.4f}")

            model.eval()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        state_dim = states.shape[1]
        action_dim = actions.shape[1]

        states_t = torch.as_tensor(states, dtype=torch.float32)
        actions_t = torch.as_tensor(actions, dtype=torch.float32)

        # atanh requires actions strictly inside (-1, 1); rescale if the dataset's
        # action bounds exceed that range
        max_abs_action = float(torch.max(torch.abs(actions_t)))
        if max_abs_action > 1.0:
            actions_t = actions_t / max_abs_action

        model = ASModel(state_dim, action_dim)
        train_AS_model(model, states_t, actions_t, epochs, batch_size, device)

        # Eq. 5: AS_f = {f_sigma(phi(s)) | s in D}; report the mean as in the paper
        with torch.no_grad():
            _, predicted_sigma = model(states_t.to(device))

        return float(predicted_sigma.mean())

    def compute_trajectory_features(episodes) -> np.ndarray:
        """Per-episode behavior descriptor: mean/std of observations, mean/std
        of actions, return, and episode length. One row per episode.

        Intended to be pooled across all dataset variants of the same
        environment (see compute_scaling_stats) and passed as
        `trajectory_scaling_stats` to compute_trajectory_diversity, so
        clustering isn't re-centered per dataset.
        """
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

        return np.array(trajectory_features)

    def compute_scaling_stats(data):
        """Per-dimension (mean, std) of `data`, for use with _standardize().

        Pool across all dataset variants of the same environment (e.g. by
        concatenating their states, or their compute_trajectory_features
        output, before calling this) so the resulting stats can be shared
        across variants instead of each one standardizing to its own
        mean/std. Without pooling, a dataset confined to a narrow region of
        state/behavior space gets rescaled to the same unit-variance spread
        as a dataset that explores widely, which defeats the point of a
        coverage/diversity metric applied by compute_state_coverage /
        compute_trajectory_diversity.
        """
        return np.mean(data, axis=0), np.std(data, axis=0)

    def _standardize(data, scaling_stats=None):
        """Z-score `data` per dimension using `scaling_stats` (mean, std), or
        data's own mean/std if `scaling_stats` is omitted (only meaningful
        when scaling a single dataset in isolation, see compute_scaling_stats)."""
        if scaling_stats is None:
            mean, std = ProfileFeatureComputer.compute_scaling_stats(data)
        else:
            mean, std = scaling_stats

        std = np.where(std == 0, 1.0, std)  # leave constant dims unscaled
        return (data - mean) / std

    def compute_cluster_model(data, n_clusters, scaling_stats=None):
        """Fit a MiniBatchKMeans with `n_clusters` clusters on standardized `data`.

        Fit once on data pooled across all dataset variants of the same
        environment (e.g. every hopper-*-v0 dataset's states, or their
        compute_trajectory_features output) and reuse the returned model via
        `.predict()` for every variant — pass it as `state_cluster_model` /
        `trajectory_cluster_model` to compute_state_coverage /
        compute_trajectory_diversity — so every variant is scored against the
        same fixed partition instead of each one fitting (and thus fully
        occupying) its own adaptive one. Without this, state_cluster_coverage
        sits at ~1.0 for every dataset regardless of how much of the space it
        actually covers, since a fresh model fit on N >> n_clusters points
        essentially never leaves a cluster empty.

        `scaling_stats` should be the same (ideally also pooled) stats passed
        to those functions for this data; see compute_scaling_stats.
        """
        scaled_data = ProfileFeatureComputer._standardize(data, scaling_stats)
        model = MiniBatchKMeans(n_clusters=n_clusters,
                                 random_state=42, n_init="auto")
        model.fit(scaled_data)
        return model

    def compute_trajectory_diversity(episodes, n_traj_clusters, trajectory_scaling_stats=None, trajectory_cluster_model=None) -> float:
        # Metric 3: Trajectory diversity
        trajectory_features = ProfileFeatureComputer.compute_trajectory_features(
            episodes)

        # Scale before clustering so high-variance features don't dominate.
        # See compute_scaling_stats for why `trajectory_scaling_stats` should
        # be pooled across sibling dataset variants.
        scaled_features = ProfileFeatureComputer._standardize(
            trajectory_features, trajectory_scaling_stats)

        # `trajectory_cluster_model`, if given, should be fit once on pooled
        # family data (see compute_cluster_model) and reused here instead of
        # fitting a fresh model per dataset.
        if trajectory_cluster_model is None:
            trajectory_cluster_model = ProfileFeatureComputer.compute_cluster_model(
                trajectory_features, n_traj_clusters, trajectory_scaling_stats)

        traj_labels = trajectory_cluster_model.predict(scaled_features)
        n_traj_clusters = trajectory_cluster_model.n_clusters

        trajectory_diversity = ProfileFeatureComputer._normalized_cluster_entropy(
            traj_labels, n_traj_clusters)

        return trajectory_diversity

    def compute_state_coverage(states, actions, n_state_clusters, state_scaling_stats=None, action_bounds=None, state_cluster_model=None) -> tuple:
        state_std = float(np.mean(np.std(states, axis=0)))

        # Scale states before clustering so high-variance dimensions don't
        # dominate the cluster assignment. See compute_scaling_stats for why
        # `state_scaling_stats` should be pooled across sibling dataset
        # variants rather than left to fit each dataset's own mean/std.
        scaled_states = ProfileFeatureComputer._standardize(
            states, state_scaling_stats)

        # `state_cluster_model`, if given, should be fit once on pooled
        # family data (see compute_cluster_model) and reused here instead of
        # fitting a fresh model per dataset — see that function's docstring
        # for why state_cluster_coverage is meaningless without this.
        if state_cluster_model is None:
            state_cluster_model = ProfileFeatureComputer.compute_cluster_model(
                states, n_state_clusters, state_scaling_stats)

        state_labels = state_cluster_model.predict(scaled_states)
        n_state_clusters = state_cluster_model.n_clusters

        state_cluster_coverage = float(
            len(np.unique(state_labels)) / n_state_clusters)
        # Entropy of the cluster-occupancy distribution from the same
        # state clustering used for state_cluster_coverage above.
        state_cluster_entropy = ProfileFeatureComputer._normalized_cluster_entropy(
            state_labels, n_state_clusters)

        # Metric 1.2: Action standard deviation and per-dimension usage
        # entropy. Unlike state_cluster_entropy this is not clustering-based:
        # it's the mean Shannon entropy of each action dimension's own value
        # histogram, i.e. how uniformly that actuator is used across its
        # observed range. `action_bounds`, if given, should be a pooled
        # per-dimension (min, max) across sibling dataset variants (see
        # compute_state_action_bounds) so every variant's histogram spans the
        # same range instead of stretching to its own min/max.
        action_std = float(np.mean(np.std(actions, axis=0)))
        action_usage_entropy = float(np.mean([
            ProfileFeatureComputer._calculate_entropy(
                actions[:, dim],
                value_range=None if action_bounds is None else (
                    action_bounds[0][dim], action_bounds[1][dim])
            ) for dim in range(actions.shape[1])
        ]))

        return state_std, state_cluster_coverage, state_cluster_entropy, action_std, action_usage_entropy
