# Dataset Analyzer

Computes a set of measurable characteristics for offline reinforcement learning
datasets.

## Metrics

For each dataset (loaded via [Minari](https://minari.farama.org/) through
`d3rlpy`), the analyzer computes:

- **Size:** number of transitions, number of episodes, average episode length.
- **Coverage:** how much of the state/action space the dataset touches:
  - `State Spread`: mean per-dimension standard deviation of observations.
  - `State Cluster Coverage`: fraction of K-Means state clusters that are
    actually occupied.
  - `State Entropy`: normalized entropy of the state-cluster occupancy
    distribution (0 = all data in one cluster, 1 = uniform across clusters).
  - `Action Variance` / `Action Entropy`: dispersion and entropy of the
    action distribution, per action dimension.
- **Quality:** statistics over per-episode returns (mean, std, min, max,
  median) and reward sparsity (fraction of zero-reward steps).
- **Diversity:** `Trajectory Diversity`: normalized entropy of episode
  clusters, where each episode is represented by summary statistics
  (mean/std of states and actions, total return, length).

Each dataset's profile is written as a JSON file to `dataset_profiles/`.
See the README in that folder for the exact schema.

## Data Sets

All the datasets are loaded via `d3rlpy` from the [Minari](https://minari.farama.org/main/) suite.

### Included in the research study:

[**MuJoCo Walker2d datasets**](https://minari.farama.org/main/datasets/mujoco/walker2d/)**:**
- [expert-v0](https://minari.farama.org/main/datasets/mujoco/walker2d/expert-v0/)
- [medium-v0](https://minari.farama.org/main/datasets/mujoco/walker2d/medium-v0/)
- [simple-v0](https://minari.farama.org/main/datasets/mujoco/walker2d/simple-v0/)

[**MuJoCo HalfCheetah datasets**](https://minari.farama.org/main/datasets/mujoco/halfcheetah/)**:**
- [expert-v0](https://minari.farama.org/main/datasets/mujoco/halfcheetah/expert-v0/)
- [medium-v0](https://minari.farama.org/main/datasets/mujoco/halfcheetah/medium-v0/)
- [simple-v0](https://minari.farama.org/main/datasets/mujoco/halfcheetah/simple-v0/)

[**MuJoCo Hopper datasets**](https://minari.farama.org/main/datasets/mujoco/hopper/)**:**
- [expert-v0](https://minari.farama.org/main/datasets/mujoco/hopper/expert-v0/)
- [medium-v0](https://minari.farama.org/main/datasets/mujoco/hopper/medium-v0/)
- [simple-v0](https://minari.farama.org/main/datasets/mujoco/hopper/simple-v0/)


### Not included in the research study:

MuJoCo Ant & Humanoid datasets are **not YET** included in this study due to their high-dimensional state and action spaces, which make them less suitable for the current analysis.

[**MuJoCo Ant datasets**](https://minari.farama.org/main/datasets/mujoco/ant/)**:**
- [expert-v0](https://minari.farama.org/main/datasets/mujoco/ant/expert-v0/)
- [medium-v0](https://minari.farama.org/main/datasets/mujoco/ant/medium-v0/)
- [simple-v0](https://minari.farama.org/main/datasets/mujoco/ant/simple-v0/)

[**MuJoCo Humanoid datasets**](https://minari.farama.org/main/datasets/mujoco/humanoid/)**:**
- [expert-v0](https://minari.farama.org/main/datasets/mujoco/humanoid/expert-v0/)
- [medium-v0](https://minari.farama.org/main/datasets/mujoco/humanoid/medium-v0/)
- [simple-v0](https://minari.farama.org/main/datasets/mujoco/humanoid/simple-v0/)