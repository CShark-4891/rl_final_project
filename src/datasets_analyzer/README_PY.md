# Dataset Analyzer Scripts

This folder contains Python utilities for analyzing offline RL datasets, building a meta-dataset, and evaluating how dataset characteristics relate to downstream performance.

## Files and purpose

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

## Typical workflow

1. Run dataset_analyzer.py to generate dataset profile JSON files.
2. Run build_meta_dataset.py to create the meta-analysis registry.
3. Use train_meta_predictor.py or predict_performance.py to study the relationship between dataset metrics and offline RL performance.
