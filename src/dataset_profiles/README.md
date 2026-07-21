# Dataset Profiles

This folder contains the output of `dataset_analyzer.py`: one JSON file per
analyzed dataset, describing its measurable characteristics.

## File naming

Each file is named after the source dataset:

```
mujoco_walker2d_medium-v0.json
```

## Schema

```json
{
    "Dataset": "mujoco/walker2d/medium-v0",

    "Size": {
        "Transitions": 1000000,
        "Episodes": 1000,
        "Average Episode Length": 1000.0
    },

    "Coverage": {
        "State Spread": 0.0,
        "State Cluster Coverage": 0.0,
        "State Entropy": 0.0,
        "Action Variance": 0.0,
        "Action Entropy": 0.0
    },

    "Quality": {
        "Mean Return": 0.0,
        "Std Return": 0.0,
        "Min Return": 0.0,
        "Max Return": 0.0,
        "Median Return": 0.0,
        "Reward Sparsity": 0.0
    },

    "Diversity": {
        "Trajectory Diversity": 0.0
    }
}
```

### Field descriptions

| Field | Meaning |
|---|---|
| `Size.Transitions` | Total number of (state, action, reward) steps across all episodes. |
| `Size.Episodes` | Number of episodes/trajectories in the dataset. |
| `Size.Average Episode Length` | Mean number of steps per episode. |
| `Coverage.State Spread` | Mean per-dimension standard deviation of the state observations. |
| `Coverage.State Cluster Coverage` | Fraction of K-Means state clusters that contain at least one sample (0–1). |
| `Coverage.State Entropy` | Normalized entropy of the state-cluster occupancy distribution (0 = concentrated, 1 = uniform). |
| `Coverage.Action Variance` | Mean variance across action dimensions. |
| `Coverage.Action Entropy` | Mean histogram-based entropy across action dimensions. |
| `Quality.Mean/Std/Min/Max/Median Return` | Statistics over per-episode summed (raw, non-normalized) reward. |
| `Quality.Reward Sparsity` | Fraction of timesteps with zero reward. |
| `Diversity.Trajectory Diversity` | Normalized entropy of episode clusters, based on per-episode summary features (state/action mean & std, return, length). |

