# Source Code overview

A concise map of the code inside the `src/` folder. Use this as a quick reference to find the main utilities and outputs.

- `configs/` : experiment configuration files (YAML).
- `dataset_analyzer/` : dataset profiling and feature-computation scripts (produces JSON profiles to `src/results/dataset_profiles/`).
- `pipeline/` : training, benchmarking and orchestration (training logs under `src/pipeline/d3rlpy_logs/`).
- `performance_prediction/` : meta-predictor training & inference code.
- `visualization/` : plotting and figure-generation utilities used for paper figures.
- `results/` : generated outputs (dataset profiles, self-trained runs: `self_trained/cql_runs/`, figures).

**Quick tips**
- Check script top-matter for CLI args and config references.
- Use `configs/` to reproduce experiments with different settings.