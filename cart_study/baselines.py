"""Single-tree baselines: ALOOF, node voting, and two-level lookahead."""

from collections import defaultdict

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from cart_study.trees import ExperimentalTree, FastPrediction, choose_min, valid_mask


def numeric_splits(x, y, min_leaf=1, criterion="gini"):
    """All distinct midpoint splits with their weighted training losses."""
    order = np.argsort(x, kind="stable")
    xs, ys = np.asarray(x)[order], np.asarray(y)[order]
    cuts = np.flatnonzero(xs[:-1] < xs[1:]) + 1
    cuts = cuts[(cuts >= min_leaf) & (len(y) - cuts >= min_leaf)]
    if not len(cuts):
        return np.array([]), np.array([])
    labels = np.unique(y)
    counts = np.cumsum(ys[:, None] == labels, axis=0)
    left = counts[cuts - 1]
    right = counts[-1] - left
    if criterion == "gini":
        loss = len(y) - (left**2 / cuts[:, None]).sum(axis=1)
        loss -= (right**2 / (len(y) - cuts[:, None])).sum(axis=1)
    elif criterion == "error":
        loss = len(y) - left.max(axis=1) - right.max(axis=1)
    else:
        raise ValueError("Unknown split criterion")
    return xs[cuts - 1] / 2 + xs[cuts] / 2, loss / len(y)


def loo_feature(x, y, min_leaf=1):
    """Exact binary LOO squared probability loss, with midpoint recomputation.

    Rows with the same value and class have the same omitted-data problem.
    Work in blocks to bound memory on continuous features. This implements the
    direct LOO definition, not the paper's specialized linear-time algorithm.
    """
    order = np.argsort(x, kind="stable")
    x, y = np.asarray(x)[order], np.asarray(y, dtype=float)[order]
    n = len(y)
    cut = np.flatnonzero(x[:-1] < x[1:]) + 1
    if not len(cut):
        return float(np.mean((y - (y.sum() - y) / (n - 1)) ** 2))
    thresholds = x[cut - 1] / 2 + x[cut] / 2
    cumulative = np.cumsum(y)[cut - 1]
    groups, first, weights = np.unique(
        np.column_stack([x, y]), axis=0, return_index=True, return_counts=True
    )
    total = 0.0
    for start in range(0, len(groups), 64):
        ids = first[start : start + 64]
        held_y = y[ids]
        was_left = ids[:, None] < cut
        nl = cut - was_left
        nr = n - 1 - nl
        pl = cumulative - was_left * held_y[:, None]
        pr = y.sum() - held_y[:, None] - pl
        with np.errstate(divide="ignore", invalid="ignore"):
            losses = (nl - pl**2 / nl - (nl - pl) ** 2 / nl) + (
                nr - pr**2 / nr - (nr - pr) ** 2 / nr
            )
        losses[(nl < min_leaf) | (nr < min_leaf)] = np.inf
        usable = np.isfinite(losses).any(axis=1)
        smallest = losses.min(axis=1)
        best = np.argmax(
            np.isclose(losses, smallest[:, None], rtol=0, atol=1e-12), axis=1
        )
        threshold = thresholds[best].copy()
        interior = (ids > 0) & (ids < n - 1)
        ii = ids[interior]
        unique = np.zeros(len(ids), dtype=bool)
        unique[interior] = (x[ii - 1] < x[ii]) & (x[ii] < x[ii + 1])
        merged = unique & ((cut[best] == ids) | (cut[best] == ids + 1))
        threshold[merged] = x[ids[merged] - 1] / 2 + x[ids[merged] + 1] / 2
        rows = np.arange(len(ids))
        with np.errstate(divide="ignore", invalid="ignore"):
            probability = np.where(
                x[ids] <= threshold,
                pl[rows, best] / nl[rows, best],
                pr[rows, best] / nr[rows, best],
            )
        probability[~usable] = (y.sum() - held_y[~usable]) / (n - 1)
        total += np.sum(weights[start : start + len(ids)] * (held_y - probability) ** 2)
    return float(total / n)


class ALOOFTree(FastPrediction, ExperimentalTree):
    """Numeric binary ALOOF with LOO probability loss and LOO no-split stopping."""

    def fit(self, X, y):
        if len(np.unique(y)) > 2:
            raise ValueError("This ALOOF implementation is binary only")
        return super().fit(X, y)

    def _choose(self, X, y):
        binary = (y == np.unique(y)[-1]).astype(float)
        candidates, scores = [], []
        for feature in range(X.shape[1]):
            thresholds, loss = numeric_splits(X[:, feature], y, self.min_samples_leaf)
            if len(thresholds):
                candidates.append((feature, thresholds[choose_min(loss)]))
                scores.append(loo_feature(X[:, feature], binary, self.min_samples_leaf))
        if not scores:
            return None
        best = choose_min(scores, self.rng_)
        parent = np.mean((binary - (binary.sum() - binary) / (len(y) - 1)) ** 2)
        return candidates[best] if scores[best] < parent - 1e-12 else None


class NodeVoteTree(FastPrediction, ExperimentalTree):
    """Dannegger-style variable voting and median thresholds at each node."""

    def __init__(self, n_subsamples=100, **kwargs):
        super().__init__(**kwargs)
        self.n_subsamples = n_subsamples

    def _choose(self, X, y):
        votes = defaultdict(list)
        for _ in range(self.n_subsamples):
            sample = self.rng_.integers(len(y), size=len(y))
            stump = DecisionTreeClassifier(
                max_depth=1,
                min_samples_leaf=self.min_samples_leaf,
                random_state=int(self.rng_.integers(2**31)),
            ).fit(X[sample], y[sample])
            feature = int(stump.tree_.feature[0])
            if feature >= 0:
                votes[feature].append(stump.tree_.threshold[0])
        if not votes:
            return None
        features = sorted(votes)
        feature = features[choose_min([-len(votes[f]) for f in features], self.rng_)]
        threshold = float(np.median(votes[feature]))
        if valid_mask(X, feature, threshold, self.min_samples_leaf) is None:
            return None
        return feature, threshold


class ErrorTree(FastPrediction, ExperimentalTree):
    """Greedy classification-error splits, with optional binary two-level search."""

    def __init__(self, lookahead=False, **kwargs):
        super().__init__(**kwargs)
        self.lookahead = lookahead

    def fit(self, X, y):
        if self.lookahead and not np.isin(X, [0, 1]).all():
            raise ValueError("Exact lookahead is restricted to binary features")
        return super().fit(X, y)

    def _grow(self, X, y, depth):
        self.remaining_ = self.max_depth - depth
        return super()._grow(X, y, depth)

    def _best_error(self, X, y):
        best = len(y) - np.unique(y, return_counts=True)[1].max()
        for feature in range(X.shape[1]):
            _, losses = numeric_splits(X[:, feature], y, self.min_samples_leaf, "error")
            if len(losses):
                best = min(best, losses.min() * len(y))
        return best

    def _choose(self, X, y):
        candidates, scores, immediate = [], [], []
        for feature in range(X.shape[1]):
            thresholds, losses = numeric_splits(
                X[:, feature], y, self.min_samples_leaf, "error"
            )
            for threshold, loss in zip(thresholds, losses):
                immediate.append(loss)
                if self.lookahead and self.remaining_ >= 2:
                    mask = X[:, feature] <= threshold
                    loss = (
                        self._best_error(X[mask], y[mask])
                        + self._best_error(X[~mask], y[~mask])
                    ) / len(y)
                candidates.append((feature, threshold))
                scores.append(loss)
        if not candidates:
            return None
        tied = np.flatnonzero(np.isclose(scores, np.min(scores), rtol=0, atol=1e-12))
        best = tied[choose_min(np.asarray(immediate)[tied], self.rng_)]
        return candidates[best]
