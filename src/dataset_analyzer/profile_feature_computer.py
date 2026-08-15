import numpy as np
import torch

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy

PRINT_VERBOSE = True


class ProfileFeatureComputer:

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

    def compute_trajectory_diversity(episodes, n_traj_clusters) -> float:
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

        trajectory_diversity = ProfileFeatureComputer.normalized_cluster_entropy(
            traj_labels, n_traj_clusters)

        return trajectory_diversity

    def compute_state_coverage(states, actions, n_state_clusters) -> tuple:
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
        state_entropy_coverage = ProfileFeatureComputer.normalized_cluster_entropy(
            state_labels, n_state_clusters)

        # Metric 1.2: Action variance and entropy -> Action Coverage
        action_variance = float(np.mean(np.var(actions, axis=0)))
        action_entropy = float(np.mean([
            ProfileFeatureComputer.calculate_entropy(actions[:, i]) for i in range(actions.shape[1])
        ]))

        return state_spread, state_cluster_coverage, state_entropy_coverage, action_variance, action_entropy
