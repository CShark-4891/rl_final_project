# Dataset Analyzer

Computes a set of measurable characteristics for offline reinforcement learning
datasets.

## Dataset Analyzer Scripts

This folder contains Python utilities for analyzing offline RL datasets, building a meta-dataset, and evaluating how dataset characteristics relate to downstream performance.

### Files and purpose

- build_meta_dataset.py
  Builds a CSV registry by combining benchmark results from the pipeline with structural dataset profile features such as coverage, entropy, diversity, and reward sparsity.

- dataset_analyzer.py
  Loads Minari datasets, computes diagnostic metrics, and saves JSON profile files describing each dataset's state/action coverage, quality, and trajectory diversity.

- generate_custom_minari.py
  Creates a local custom Minari dataset for experimentation, useful for validating pipeline behavior on synthetic or custom replay-style data.

- train_meta_predictor.py
  Trains a simple meta-model on the compiled registry to estimate which dataset properties are most predictive of performance.

- predict_performance.py
  Uses a trained meta-predictor to estimate performance for a dataset profile and compares those predictions with real pipeline results.

### Typical workflow

1. Run dataset_analyzer.py to generate dataset profile JSON files.
2. Run build_meta_dataset.py to create the meta-analysis registry.
3. Use train_meta_predictor.py or predict_performance.py to study the relationship between dataset metrics and offline RL performance.

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