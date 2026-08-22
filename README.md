# Analyzing Dataset Characteristics for Offline Reinforcement Learning with CQL

This repository studies which measurable characteristics of offline reinforcement learning datasets correlate with downstream policy performance. We compute a set of dataset metrics, build a meta-dataset aggregating results, and evaluate how well those metrics predict performance for Conservative Q-Learning (CQL) across D4RL and Minari datasets.

**Key contributions:**
- Computation of coverage, diversity and quality metrics for offline RL datasets (state/action std, cluster coverage & entropy, action usage entropy, trajectory diversity, reward sparsity).
- Aggregation pipeline to combine dataset profiles with benchmarked model results into a meta-dataset for analysis.
- Experiments training CQL (via d3rlpy) on Minari datasets and comparing metric–performance relationships against published D4RL/CQL results.

**Paper:** The full report and findings are in the repo (LaTeX source). See the project paper for details, figures and references.

**Quick links:**
- Dataset analysis: [src/dataset_analyzer](src/dataset_analyzer)
- Dataset profile helpers: [src/dataset_analyzer/profile_feature_computer.py](src/dataset_analyzer/profile_feature_computer.py)
- Training & benchmarking: [src/pipeline/train.py](src/pipeline/train.py) and [src/pipeline/benchmark.py](src/pipeline/benchmark.py)
- Meta-prediction (proof of concept): [src/performance_prediction](src/performance_prediction)
- Results and profiles: [src/results](src/results)

**Repository layout (top-level)**
- `literature/` : contains the pdf of different papers that we analyzed for our study.
- `reports/` : contains the PDF of the project report.
- `src/` : code for dataset analysis, training pipelines, predictors and visualization.
- `src/dataset_analyzer/` : compute per-dataset JSON profiles and helper utilities.
- `src/pipeline/` : training and benchmarking orchestration (CQL runs via d3rlpy).
- `src/performance_prediction/` : scripts to train/predict model performance from dataset features.
- `src/results/` : computed `dataset_profiles/`, trained-policy results, and figures.
- `figures/`, `result_plots/` : plotting outputs used in the paper.
- `references.bib` : BibTeX references for the project and underlying works.

**Quickstart (conda / venv)**
1. Create environment and install deps (requirements are in the repo):

```bash
python -m venv .venv
source .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -e .
pip install -r src/requirements.txt
```
2. Compute dataset profiles (writes JSON profiles to `src/results/dataset_profiles/`):
```bash
python src/dataset_analyzer/dataset_analyzer.py 
```


3. Build the meta-dataset CSV used for analysis:

```bash
python src/dataset_analyzer/build_meta_dataset.py 
```

4. Train CQL experiments with the pipeline (writes trained models and metrics to `src/results/self_trained/cql_runs/` and logs to `src/pipeline/d3rlpy_logs/`):

```bash
python src/pipeline/main.py
```

5. Generate figures from the results and profiles:

```bash
python src/visualization/visualize_results.py
```
See the individual script docs and `configs/` for additional options and hyperparameters.

**Where results live**
- Computed dataset profiles: [src/results/dataset_profiles](src/results/dataset_profiles)
- Trained runs & logs: [src/results/self_trained/cql_runs](src/results/self_trained/cql_runs)
- Figures used in the report: [src/results/figures](src/results/figures)

**Main findings (summary)**
- Trajectory Diversity (clustering-based episode diversity) is the strongest single predictor of final CQL performance in our analyses.
- Coverage metrics (state/action std, cluster coverage/entropy, action usage entropy) provide useful signals but can be sensitive to family-pooled normalization and chosen clustering resolution.
- Return-based metrics from prior work (ERI, TQ) distinguish some tiers but are not uniformly reliable across D4RL vs Minari sources.
- Differences between D4RL and Minari datasets mean they are not always directly interchangeable; dataset source and collection procedure influence metric distributions.

**Limitations & recommendations**
- Our reproduced CQL runs used fewer training steps and seeds than some reference papers; reproduce results may require matching those training volumes and hyperparameters exactly.
- Many clustering-based metrics depend on the chosen number of clusters and family-pooled preprocessing. Check `profile_feature_computer.py` for implementation details.

**Reproduce the paper figures**
1. Compute profiles and build the registry (steps above).
2. Re-run benchmarks or use the precomputed runs in `src/results/self_trained/cql_runs/`.
3. Generate figures with the visualization script `src/visualization/visualize_results.py`.

**Cite / Acknowledge**
If you use this code or results, please cite the project and the underlying works (CQL, D4RL, d3rlpy, Minari) referenced in the paper. Also see the [references section](#references-of-the-project) below for links to the original papers and repositories.

**Project Authors:**
Alex Alfonso Trigo, Luca Beuke, Simon Böke

## References of the project

### Papers and Documentations:

- [Conservative Q-Learning (CQL) - Paper](https://arxiv.org/abs/2006.04779)

- [D3rlpy: Deep Reinforcement Learning Library for Python - Paper](https://arxiv.org/abs/2111.03788)

- [D4RL: Datasets for Deep Data-Driven Reinforcement Learning - Paper](https://arxiv.org/abs/2004.07219)

- [Minari: A Benchmark for Offline Reinforcement Learning - Documentation](https://minari.farama.org/main/)

- [Understanding the Effects of Dataset Characteristics on Offline Reinforcement Learning - Paper](https://arxiv.org/abs/2111.04714v1)

- [Measuring Data Quality for Dataset Selection in Offline Reinforcement Learning - Paper](https://arxiv.org/abs/2111.13461)

### Repositories:

- [D3rlpy: Deep Reinforcement Learning Library for Python - Github](https://github.com/takuseno/d3rlpy)

- [D4RL: Datasets for Deep Data-Driven Reinforcement Learning - Github](https://github.com/Farama-Foundation/D4RL)

- [Minari: A Benchmark for Offline Reinforcement Learning - Github](https://github.com/Minari/Minari)

