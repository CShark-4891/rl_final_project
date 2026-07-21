# Configuration Specifications

This directory stores reproducible YAML or JSON hyperparameter files for the CQL algorithm.

## Notice
To protect the integrity of the experiment, algorithmic parameters must remain completely static during comparisons. This folder separates operational controls (e.g., batch size, learning rates, conservative alpha scaling factors) from execution scripts.

*   `cql_default.yaml`: Houses the hyperparameter set used for all experiments. This file is referenced by the `train.py` script and should not be modified during the course of this study/work.