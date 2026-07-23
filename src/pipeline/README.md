# Pipeline

This folder contains the scripts that train, evaluate, and benchmark the offline reinforcement learning experiments.

## Scripts

### `train.py`
Trains a CQL agent on a single dataset split with a fixed random seed.

- Loads the experiment configuration from `configs/cql_default.yaml`.
- Loads the Minari dataset through `d3rlpy`.
- Creates the CQL learner with the configured optimizer and regularization settings.
- Runs offline training for the requested number of gradient steps.
- Saves the trained model as a `.d3` artifact.

### `benchmark.py`
Evaluates a saved model inside the corresponding Gymnasium environment.

- Loads the trained `.d3` model.
- Recreates the matching Gymnasium environment for the dataset family.
- Runs a fixed number of evaluation episodes.
- Computes the mean raw return, return standard deviation, and D4RL-normalized score.
- Writes the evaluation summary to `metrics.json`.

### `main.py`
Orchestrates the full experiment loop across datasets and seeds.

- Iterates over the configured MuJoCo dataset list.
- Builds the result directory for each dataset and seed combination.
- Runs `train.py` first and then `benchmark.py` for the same model.
- Skips work when the expected model or metrics file already exists.
- Aggregates all seed-level metrics into `global_pipeline_report.json`.

## Outputs

Each run writes its artifacts under `results/cql_runs/`.

- `model.d3`: trained policy checkpoint.
- `metrics.json`: evaluation metrics for one seed.
- `global_pipeline_report.json`: combined summary across all seeds for one dataset.

## Notes

- The pipeline is designed for reproducible offline RL experiments.
- The active dataset list is defined in `main.py`.
- Default hyperparameters come from the shared configuration file in `configs/`.


