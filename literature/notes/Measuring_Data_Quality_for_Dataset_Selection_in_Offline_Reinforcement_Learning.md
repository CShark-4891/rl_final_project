# Notes on Measuring Data Quality for Dataset Selection in Offline Reinforcement Learning

offline RL performance depends heavily on the quality of the dataset

problem:

- practitioners often have multiple datasets available
- training every offline RL algorithm on every dataset is expensive
- no simple way to estimate which dataset is most useful

main contribution:

- proposes simple metrics to estimate dataset quality before training
- introduces:
  - Estimated Relative Improvement (ERI)
  - Estimated Action Stochasticity (EAS)
  - Combined Indicator (COI)
- demonstrates that these metrics correlate well with offline RL performance on benchmark datasets

---

## Main content and ideas

### Introduction

- offline RL enables learning from previously collected datasets without interacting with the environment
- algorithm performance depends strongly on the dataset quality
- in practice many datasets may exist, but training an offline RL algorithm on each one is expensive
- unlike supervised learning, dataset selection for offline RL has received little attention
- goal: estimate whether a dataset is worth using before investing computational resources into training

---

### Related Work

- reviews major offline RL algorithms (BCQ, CQL, BEAR, etc.)
- existing work focuses on designing better learning algorithms
- almost no work studies *which dataset* should be selected
- draws inspiration from data quality research in databases and supervised learning

---

### Problem Statement

Given several offline datasets:

- which one is most promising for offline RL?
- cannot simply compare average returns
- datasets should be judged by whether they allow improvement over the recorded behavior

desired indicator should

- be inexpensive
- require no policy training
- correlate with final offline RL performance

---

### Proposed Data Quality Indicators

The paper proposes two simple measures.

#### Estimated Relative Improvement (ERI)

idea:

- compare the best trajectories to the average trajectories
- if the best trajectories are much better than the average, there is room for offline RL to improve upon the behavior policy

high ERI

- large improvement potential
- dataset contains successful demonstrations that algorithms can learn from

low ERI

- dataset already near optimal
- or all trajectories are similarly poor
- limited opportunity for improvement

---

#### Estimated Action Stochasticity (EAS)

idea:

- estimate how deterministic the behavior policy is

high stochasticity

- same states lead to many different actions
- behavior policy is inconsistent
- learning becomes harder

low stochasticity

- actions are more predictable
- easier for offline RL to model the behavior policy
- usually associated with better datasets

---

#### Combined Indicator (COI)

combines

- ERI
- EAS

goal

- datasets should have
  - high improvement potential
  - low action randomness

COI provides a single score for ranking candidate datasets

---

### Experiments

- evaluate indicators on common D4RL benchmark datasets
- compare rankings produced by the indicators against actual offline RL performance
- test several offline RL algorithms rather than a single method

results

- ERI alone correlates well with achievable improvement
- EAS captures an independent aspect of dataset quality
- COI generally gives the most reliable ranking across datasets
- simple metrics are surprisingly effective despite requiring almost no computation

---

### Limitations

- indicators are heuristics, not guarantees
- performance still depends on
  - chosen offline RL algorithm
  - hyperparameters
  - task characteristics
- paper mainly evaluates on benchmark datasets rather than large industrial datasets
- dataset quality is more complex than the proposed metrics capture

---

## Key Takeaways

- offline RL success depends as much on the dataset as on the algorithm
- practitioners should perform dataset selection before training
- proposed metrics:
  - **ERI** → estimates improvement potential
  - **EAS** → estimates consistency of the behavior policy
  - **COI** → combines both into a practical dataset quality score
- simple quality indicators can save significant computation by identifying promising datasets early
