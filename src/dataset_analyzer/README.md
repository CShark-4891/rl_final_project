# Dataset Analyzer

Computes a set of measurable characteristics for offline reinforcement learning
datasets.

## Overview

This package contains utilities for computing descriptive statistics and diagnostic features for offline RL datasets, producing per-dataset JSON profiles and a combined meta-dataset used by downstream analysis and prediction models.

## Scripts and locations

- `dataset_analyzer.py` (this folder): load datasets via `d3rlpy`/Minari and compute per-dataset JSON profiles describing size, coverage, quality, and trajectory diversity.
- `build_meta_dataset.py` (this folder): aggregate per-run benchmark results with the computed dataset profiles to produce a CSV registry for meta-analysis.
- `profile_feature_computer.py` (this folder): lower-level helpers used by the analyzer to compute coverage, entropy, histograms and pooled-family statistics.
- `generate_custom_minari.py` (this folder): helper to create local Minari-style datasets for experiments.
- `train_meta_predictor.py` and `predict_performance.py` are located in `src/performance_prediction/`: training and inference code for simple meta-models that predict downstream algorithm performance from dataset profiles.

## Typical workflow

1. Run `dataset_analyzer.py` to produce per-dataset JSON profiles (outputs are written to `src/results/dataset_profiles/`).
2. Run `build_meta_dataset.py` to combine pipeline benchmark results with those profiles into a single CSV registry for modeling and visualization.
3. Train or run meta-predictors with the scripts in `src/performance_prediction/` to study which dataset features predict algorithm performance.

## Metrics

For each dataset (loaded via [Minari](https://minari.farama.org/) through
`d3rlpy`), the analyzer computes:

- **Size:** number of transitions, number of episodes, average episode length.
- **Coverage:** how much of the state/action space the dataset touches:
  - `State Standard Deviation`: mean per-dimension standard deviation of
    observations.
  - `State Cluster Coverage`: fraction of K-Means state clusters that are
    actually occupied.
  - `State Cluster Entropy`: normalized entropy of the state-cluster
    occupancy distribution from that same clustering (0 = all data in one
    cluster, 1 = uniform across clusters).
  - `Action Standard Deviation`: mean per-dimension standard deviation of
    actions — same statistic as `State Standard Deviation`, applied to
    actions instead of observations.
  - `Action Usage Entropy`: mean Shannon entropy of each action dimension's
    own value histogram — how uniformly that actuator is used across its
    observed range. Unlike `State Cluster Entropy`, this is not
    clustering-based.
- **Quality:** statistics over per-episode returns (mean, std, min, max,
  median) and reward sparsity (fraction of zero-reward steps).
- **Diversity:** `Trajectory Diversity`: normalized entropy of episode
  clusters, where each episode is represented by summary statistics
  (mean/std of states and actions, total return, length).

`State Cluster Coverage`, `State Cluster Entropy`, `Action Usage Entropy`,
and `Trajectory Diversity` are all computed against a shared reference
pooled across all sibling variants of the same environment (e.g. every
hopper-*-v0 dataset), via `compute_family_pooled_stats` in
`dataset_analyzer.py`. Concretely: the state and trajectory K-Means models
are fit once on data pooled across the whole family and reused (via
`.predict()`) for every variant, instead of each dataset fitting its own
model; the scaling/histogram-range statistics are pooled the same way. This
mirrors the bounds-pooling `dataset_analyzer.py` already does for SACo, and
for the same reason: without a shared reference, a narrow dataset (e.g.
expert) gets rescaled/rebinned/clustered to fill the same range as a wide one
(e.g. random), making it look just as covering/diverse even though it
explores far less of the space. In particular, `State Cluster Coverage` used
to sit at ~1.0 for every dataset (fitting 20 clusters fresh on hundreds of
thousands of that dataset's own points essentially never leaves a cluster
empty); scored against the family's shared 20-cluster partition instead, a
narrow dataset now only occupies the handful of clusters its points actually
fall into.

Each dataset's profile is written as a JSON file to `src/results/dataset_profiles/`.
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

MuJoCo Ant & Humanoid datasets are not included in this study due to limited time constraints.

[**MuJoCo Ant datasets**](https://minari.farama.org/main/datasets/mujoco/ant/)**:**
- [expert-v0](https://minari.farama.org/main/datasets/mujoco/ant/expert-v0/)
- [medium-v0](https://minari.farama.org/main/datasets/mujoco/ant/medium-v0/)
- [simple-v0](https://minari.farama.org/main/datasets/mujoco/ant/simple-v0/)

[**MuJoCo Humanoid datasets**](https://minari.farama.org/main/datasets/mujoco/humanoid/)**:**
- [expert-v0](https://minari.farama.org/main/datasets/mujoco/humanoid/expert-v0/)
- [medium-v0](https://minari.farama.org/main/datasets/mujoco/humanoid/medium-v0/)
- [simple-v0](https://minari.farama.org/main/datasets/mujoco/humanoid/simple-v0/)


## References

- [Minari: Offline Reinforcement Learning Datasets - Documentation](https://minari.farama.org/main/)

- [Minari: Offline Reinforcement Learning Datasets - Github](https://github.com/Farama-Foundation/Minari)

- [D4RL: Datasets for Deep Data-Driven Reinforcement Learning - Paper](https://arxiv.org/abs/2004.07219)

- [D4RL: Datasets for Deep Data-Driven Reinforcement Learning - Github](https://github.com/Farama-Foundation/D4RL)