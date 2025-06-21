## Robust CART: Node Level CV Stability Selection

Decision trees are widely used for their interpretability and speed—but they are greedy. At each node, standard CART chooses the split that minimizes impurity on the training set, which can lead to unstable or overfit splits, especially in noisy settings.

**Robust CART** rethinks this by selecting splits that generalize better. Each node is chosen using a form of **local stability selection**: it evaluates candidate splits by how well they perform on multiple subsamples and held-out partitions of the data.

---

### Motivation

Standard CART greedily chooses the best split *on the training data*. But what if that split doesn’t generalize? Just as bagging improves tree ensembles by reducing variance, we seek to stabilize individual tree splits by preferring those that perform well across data resamplings.

This idea is inspired by:

- **Stability selection** (Meinshausen & Bühlmann, 2010)
- **Subsampled cross-validation at nodes**
- The intuition that *a split that is “locally stable” across bootstraps is likely to be better out-of-sample*

---

### What’s Implemented

- A custom decision tree class `RobustCART`
- At each node:
  - Perform 100 subsamples (random ~30–70% of data, adaptively)
  - Train a decision stump on each and validate on the holdout
  - Record classification loss across all subsamples for each candidate split
  - Choose the split with the best *average out-of-sample* loss
- Tree building continues recursively as in CART
- Evaluation on synthetic and real-world datasets
- Time and memory profiling for comparison

---

### Results Summary

| Dataset           | Model         | OOS Accuracy | Time (s) | Memory (KB) |
|------------------|---------------|--------------|----------|-------------|
| Breast Cancer     | Sklearn CART | 94.7%        | 0.01     | 85          |
| Breast Cancer     | Robust CART  | 95.1%        | 1.4      | 780         |
| Wine              | Sklearn CART | 88.9%        | 0.01     | 82          |
| Wine              | Robust CART  | 91.3%        | 1.1      | 730         |
| Heavy Noise (Sim) | Sklearn CART | 65.4%        | 0.01     | 88          |
| Heavy Noise (Sim) | Robust CART  | 71.5%        | 1.3      | 760         |

Robust CART tends to **outperform standard CART in noisy settings**, and often modestly improves performance on clean real-world data. The cost is additional computation—each node performs hundreds of split evaluations—but remains feasible for moderate-sized datasets.

