# Influence of Dataset Characteristic Combinations on Offline RL

This repository contains the codebase for investigating how combinations of measurable dataset characteristics influence the performance of Conservative Q-Learning (CQL) in offline Reinforcement Learning. Moving away from algorithmic modifications or custom data generation, this project treats the dataset as the primary variable. We leverage standardized D4RL/Minari datasets to discover which structural signatures within static data buffers correlate with successful policy deployment.

## Research Core & Hypothesis
Rather than isolating individual dataset properties (such as size or average return) or brute-forcing algorithmic configurations, this study focuses on **metric interactions**. We analyze how spatial coverage, trajectory quality distributions, and action stochasticity compound to create ideal learning conditions or trigger distribution shifts for out-of-distribution (OOD) actions.

## System Architecture

TODO insert a new diagram since the old is outdated and the new one is not yet created.

## Getting Started

Clone the repository and install dependencies with python

TODO insert a better installation guide here.


## Implementation Roadmap

To ensure a structured, hypothesis-driven workflow over the 1–2 week full-time scope, implementation is broken down into four distinct, trackable phases.

### Phase 1: Dataset Analysis & Profiling
[X] Implement `dataset_analyzer.py` to compute measurable characteristics for each dataset.
[X] Store the output as JSON files in `dataset_profiles/` for later reference.


## Phase 2: CQL Training & Evaluation Pipeline

[X] Implement `train.py` to train a CQL agent on a single dataset split with a fixed random seed.
[X] Implement `benchmark.py` to evaluate a saved model inside the corresponding Gymnasium environment.
[X] Implement `main.py` to orchestrate the full experiment loop across datasets and seeds, aggregating metrics into `global_pipeline_report.json`.
[ ] Ensure reproducibility and logging of all artifacts under `results/cql_runs/`.

## Phase 3: Result Analysis & Visualization

[ ] Implement scripts to generate plots and analytics from the aggregated metrics.
[ ] Explore correlations between dataset characteristics and CQL performance.
[ ] Identify interaction effects and potential distribution shifts.
[ ] Optional: Build a predictive model to estimate CQL performance based on dataset metrics.

## Phase 4: Documentation & Reporting

[ ] Compile findings into a comprehensive report.
[ ] Document the codebase and provide usage instructions.
[ ] Prepare for potential publication or presentation of results.