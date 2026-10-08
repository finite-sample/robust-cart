"""Small reference implementations for the notebook experiments.

These are classification experiments, not a scikit-learn estimator package.
All methods accept finite numeric features and arbitrary discrete class labels.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier


@dataclass
class Node:
    """One prediction leaf or feature-threshold branch."""

    prediction: object
    count: int
    feature: int = -1
    threshold: float = 0.0
    left: object = None
    right: object = None


def majority(y):
    """Resolve ties using sorted class order, like the CART baseline."""
    labels, counts = np.unique(y, return_counts=True)
    return labels[np.argmax(counts)]


def valid_mask(X, feature, threshold, min_leaf):
    mask = X[:, feature] <= threshold
    if min(mask.sum(), (~mask).sum()) < min_leaf:
        return None
    return mask


def split_loss(X, y, feature, threshold):
    """Half the weighted child Gini; agrees with p(1-p) for binary data."""
    mask = X[:, feature] <= threshold
    loss = 0.0
    for side in (mask, ~mask):
        if not side.any():
            continue
        _, counts = np.unique(y[side], return_counts=True)
        loss += side.mean() * (1 - np.sum((counts / counts.sum()) ** 2)) / 2
    return float(loss)


def fixed_split_predict(Xtrain, ytrain, Xtest, feature, threshold):
    """Estimate leaf labels on training rows only, then predict test rows."""
    mask = Xtrain[:, feature] <= threshold
    left = majority(ytrain[mask]) if mask.any() else majority(ytrain)
    right = majority(ytrain[~mask]) if (~mask).any() else majority(ytrain)
    return np.where(Xtest[:, feature] <= threshold, left, right)


class ExperimentalTree:
    """Shared routing and stopping rules for the custom single trees."""

    def __init__(self, max_depth=5, min_samples_leaf=5, random_state=0):
        if max_depth < 0 or min_samples_leaf < 1:
            raise ValueError("Invalid tree size limits")
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or not len(y):
            raise ValueError("Expected a nonempty matrix and matching label vector")
        if not np.isfinite(X).all():
            raise ValueError("Features must be finite")
        self.n_features_in_ = X.shape[1]
        self.rng_ = np.random.default_rng(self.random_state)
        self.diagnostics_ = []
        self.tree_ = self._grow(X, y, 0)
        return self

    def _grow(self, X, y, depth):
        node = Node(majority(y), len(y))
        if (
            depth >= self.max_depth
            or len(y) < 2 * self.min_samples_leaf
            or len(np.unique(y)) == 1
        ):
            return node
        split = self._choose(X, y)
        if split is None:
            return node
        feature, threshold = split
        mask = valid_mask(X, feature, threshold, self.min_samples_leaf)
        if mask is None:
            return node
        node.feature, node.threshold = int(feature), float(threshold)
        node.left = self._grow(X[mask], y[mask], depth + 1)
        node.right = self._grow(X[~mask], y[~mask], depth + 1)
        return node

    def predict(self, X):
        if not hasattr(self, "tree_"):
            raise ValueError("Fit the tree before prediction")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Prediction features do not match training features")
        if not np.isfinite(X).all():
            raise ValueError("Features must be finite")
        predictions = []
        for row in X:
            node = self.tree_
            while node.feature >= 0:
                node = node.left if row[node.feature] <= node.threshold else node.right
            predictions.append(node.prediction)
        return np.asarray(predictions)


class HoldoutTree(ExperimentalTree):
    """Original selected-stump scoring, or a separate common-fold variant.

    ``selected`` averages losses only when a stump proposes that exact split.
    ``common`` reserves half the node for proposal generation, then scores all
    proposed splits on the same three folds of the remaining observations.
    Proposals never use the labels of those validation observations. Leaf labels
    for each fold use proposal rows plus the other evaluation folds.
    """

    def __init__(self, scoring="selected", n_subsamples=100, **kwargs):
        super().__init__(**kwargs)
        if scoring not in {"selected", "common"} or n_subsamples < 1:
            raise ValueError("Invalid holdout settings")
        self.scoring = scoring
        self.n_subsamples = n_subsamples

    def _proposals(self, X, y):
        n = len(y)
        size = max(1, min(n - 1, int(n * min(0.7, max(0.3, 200 / n)))))
        for _ in range(self.n_subsamples):
            train = self.rng_.choice(n, size, replace=False)
            test = np.setdiff1d(np.arange(n), train)
            if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
                continue
            stump = DecisionTreeClassifier(
                max_depth=1, random_state=int(self.rng_.integers(2**31))
            ).fit(X[train], y[train])
            if stump.tree_.feature[0] >= 0:
                yield (
                    (int(stump.tree_.feature[0]), float(stump.tree_.threshold[0])),
                    float(np.mean(stump.predict(X[test]) != y[test])),
                )

    def _choose(self, X, y):
        scores = defaultdict(list)
        if self.scoring == "selected":
            for split, loss in self._proposals(X, y):
                if valid_mask(X, *split, self.min_samples_leaf) is not None:
                    scores[split].append(loss)
        else:
            proposal, evaluation = np.array_split(self.rng_.permutation(len(y)), 2)
            if len(evaluation) < 2:
                return None
            candidates = sorted(
                {s for s, _ in self._proposals(X[proposal], y[proposal])}
            )
            candidates = [
                s
                for s in candidates
                if valid_mask(X, *s, self.min_samples_leaf) is not None
            ]
            folds = KFold(
                n_splits=min(3, len(evaluation)),
                shuffle=True,
                random_state=int(self.rng_.integers(2**31)),
            )
            for train, test in folds.split(evaluation):
                fitting = np.concatenate([proposal, evaluation[train]])
                testing = evaluation[test]
                for split in candidates:
                    prediction = fixed_split_predict(
                        X[fitting], y[fitting], X[testing], *split
                    )
                    scores[split].extend((prediction != y[testing]).tolist())
        if not scores:
            return None
        best = min(scores, key=lambda s: (np.mean(scores[s]), s))
        self.diagnostics_.append(
            {
                "candidates": len(scores),
                "min_evaluations": min(map(len, scores.values())),
                "max_evaluations": max(map(len, scores.values())),
                "selected_evaluations": len(scores[best]),
            }
        )
        return best


def split_set(model):
    """Distinct feature-threshold pairs; excludes order and leaf predictions."""
    tree = model.tree_
    if isinstance(tree, Node):
        stack, splits = [tree], set()
        while stack:
            node = stack.pop()
            if node.feature >= 0:
                splits.add((node.feature, node.threshold))
                stack.extend([node.left, node.right])
        return splits
    return {(int(f), float(t)) for f, t in zip(tree.feature, tree.threshold) if f >= 0}


def root_feature(model):
    tree = model.tree_
    return int(tree.feature if isinstance(tree, Node) else tree.feature[0])


def leaf_count(model):
    if not isinstance(model.tree_, Node):
        return int(model.get_n_leaves())
    stack, count = [model.tree_], 0
    while stack:
        node = stack.pop()
        if node.feature < 0:
            count += 1
        else:
            stack.extend([node.left, node.right])
    return count


def split_support(split_sets, weights=None):
    """One vote per tree per split, including empty trees in the denominator."""
    if not split_sets:
        raise ValueError("Need at least one tree")
    weights = np.ones(len(split_sets)) if weights is None else np.asarray(weights)
    if (
        weights.shape != (len(split_sets),)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or weights.sum() <= 0
    ):
        raise ValueError("Invalid tree weights")
    counts = Counter()
    for splits, weight in zip(split_sets, weights):
        for split in set(splits):
            counts[split] += float(weight)
    return {split: count / weights.sum() for split, count in counts.items()}


class ConsensusTree(ExperimentalTree):
    """Build a real tree from a pool's global split support.

    Rank admissible candidates by support * exp(-eta * child_loss). For eta=0
    this is frequency selection. This is a single exponential reweighting, not
    an online multiplicative-weights algorithm or a Mann-Whitney statistic.
    """

    def __init__(self, support, tau=0.5, eta=0.0, **kwargs):
        super().__init__(**kwargs)
        if not 0 <= tau <= 1 or eta < 0:
            raise ValueError("Invalid consensus settings")
        self.support = support
        self.tau = tau
        self.eta = eta

    def _choose(self, X, y):
        ranked = []
        for split, frequency in sorted(self.support.items()):
            if frequency < self.tau:
                continue
            if valid_mask(X, *split, self.min_samples_leaf) is None:
                continue
            score = frequency * np.exp(-self.eta * split_loss(X, y, *split))
            ranked.append((-score, split))
        return min(ranked)[1] if ranked else None


def resample_indices(n, kind, rng):
    if kind == "bootstrap":
        return rng.choice(n, n, replace=True)
    if kind == "subsample":
        return rng.choice(n, max(1, int(0.8 * n)), replace=False)
    raise ValueError("Unknown resampling kind")


@dataclass
class Pool:
    """Resampled trees and errors from rows each individual tree did not fit."""

    trees: list
    errors: np.ndarray
    samples: list
    oob: list

    @property
    def split_sets(self):
        return [split_set(tree) for tree in self.trees]


def fit_pool(X, y, kind="bootstrap", B=20, seed=0, **tree_options):
    rng = np.random.default_rng(seed)
    trees, errors, samples, heldout = [], [], [], []
    if len(y) < 2 or B < 1:
        raise ValueError("A pool needs at least two rows and one tree")
    for _ in range(B):
        for attempt in range(100):
            sample = resample_indices(len(y), kind, rng)
            oob = np.setdiff1d(np.arange(len(y)), sample)
            if len(oob):
                break
        else:
            raise ValueError("Could not draw a nonempty out-of-bag set")
        tree = DecisionTreeClassifier(
            random_state=int(rng.integers(2**31)), **tree_options
        ).fit(X[sample], y[sample])
        trees.append(tree)
        errors.append(np.mean(tree.predict(X[oob]) != y[oob]))
        samples.append(sample)
        heldout.append(oob)
    return Pool(trees, np.asarray(errors), samples, heldout)


def mean_jaccard(sets):
    """Empty/empty is undefined and excluded; report its pair count separately."""
    values, empty_pairs = [], 0
    for a, b in combinations(sets, 2):
        if not (a or b):
            empty_pairs += 1
        else:
            values.append(len(a & b) / len(a | b))
    return (float(np.mean(values)) if values else np.nan), empty_pairs


def prediction_disagreement(predictions):
    """Average disagreement over distinct refit pairs and fixed test cases."""
    predictions = np.asarray(predictions)
    if predictions.ndim != 2 or len(predictions) < 2:
        raise ValueError("Need at least two refits on the same test cases")
    return float(np.mean([np.mean(a != b) for a, b in combinations(predictions, 2)]))


def root_agreement(roots):
    """Pairwise agreement on root feature; -1 denotes an unsplit tree."""
    return float(np.mean([a == b for a, b in combinations(roots, 2)]))


def development_split(y, seed):
    """Split original row identities before any resampling."""
    return train_test_split(
        np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed
    )


def cv_folds(y, seed):
    """Fold generation is called on distinct original observations."""
    _, counts = np.unique(y, return_counts=True)
    n_splits = min(3, int(counts.min()))
    if n_splits < 2:
        raise ValueError("Need at least two observations per class for tuning")
    return list(
        StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(
            np.zeros(len(y)), y
        )
    )


def choose_min(values, rng=None):
    values = np.asarray(values)
    tied = np.flatnonzero(np.isclose(values, values.min(), rtol=0, atol=1e-12))
    return int(tied[0] if rng is None else rng.choice(tied))


class FastPrediction:
    """Route batches through the same fitted tree instead of looping over rows."""

    def fit(self, X, y):
        self.prediction_dtype_ = np.asarray(y).dtype
        return super().fit(X, y)

    def predict(self, X):
        if not hasattr(self, "tree_"):
            raise ValueError("Fit the tree before prediction")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Prediction features do not match training features")
        if not np.isfinite(X).all():
            raise ValueError("Features must be finite")
        prediction = np.empty(len(X), dtype=self.prediction_dtype_)
        stack = [(self.tree_, np.arange(len(X)))]
        while stack:
            node, ids = stack.pop()
            if node.feature < 0:
                prediction[ids] = node.prediction
            else:
                left = X[ids, node.feature] <= node.threshold
                stack.extend([(node.left, ids[left]), (node.right, ids[~left])])
        return prediction


class HoldoutAblationTree(FastPrediction, HoldoutTree):
    """Original scoring with separately switchable tie and label-filter changes."""

    def __init__(self, random_ties=False, keep_single_class=False, **kwargs):
        super().__init__(**kwargs)
        self.random_ties = random_ties
        self.keep_single_class = keep_single_class

    def _proposals(self, X, y):
        if not self.keep_single_class:
            yield from super()._proposals(X, y)
            return
        n = len(y)
        if n < 2:
            return
        size = max(1, min(n - 1, int(n * min(0.7, max(0.3, 200 / n)))))
        for _ in range(self.n_subsamples):
            train = self.rng_.choice(n, size, replace=False)
            test = np.setdiff1d(np.arange(n), train)
            if len(np.unique(y[train])) < 2:
                continue
            stump = DecisionTreeClassifier(
                max_depth=1, random_state=int(self.rng_.integers(2**31))
            ).fit(X[train], y[train])
            if stump.tree_.feature[0] >= 0:
                yield (
                    (int(stump.tree_.feature[0]), float(stump.tree_.threshold[0])),
                    float(np.mean(stump.predict(X[test]) != y[test])),
                )

    def _choose(self, X, y):
        if not self.random_ties:
            return super()._choose(X, y)
        scores = defaultdict(list)
        if self.scoring == "selected":
            for split, loss in self._proposals(X, y):
                if valid_mask(X, *split, self.min_samples_leaf) is not None:
                    scores[split].append(loss)
        else:
            proposal, evaluation = np.array_split(self.rng_.permutation(len(y)), 2)
            if len(evaluation) < 2:
                return None
            candidates = sorted(
                {s for s, _ in self._proposals(X[proposal], y[proposal])}
            )
            candidates = [
                s
                for s in candidates
                if valid_mask(X, *s, self.min_samples_leaf) is not None
            ]
            folds = KFold(
                min(3, len(evaluation)),
                shuffle=True,
                random_state=int(self.rng_.integers(2**31)),
            )
            for train, test in folds.split(evaluation):
                fitting = np.concatenate([proposal, evaluation[train]])
                for split in candidates:
                    pred = fixed_split_predict(
                        X[fitting], y[fitting], X[evaluation[test]], *split
                    )
                    scores[split].extend((pred != y[evaluation[test]]).tolist())
        if not scores:
            return None
        candidates = sorted(scores)
        losses = [np.mean(scores[s]) for s in candidates]
        return candidates[choose_min(losses, self.rng_)]


class VectorizedConsensusTree(FastPrediction, ConsensusTree):
    def __init__(self, *args, random_ties=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_ties = random_ties

    def _choose(self, X, y):
        candidates, frequencies = [], []
        for split, support in sorted(self.support.items()):
            if support < self.tau:
                continue
            candidates.append(split)
            frequencies.append(support)
        if not candidates:
            return None
        features, thresholds = np.array(candidates).T
        masks = X[:, features.astype(int)] <= thresholds
        nl = masks.sum(axis=0)
        nr = len(y) - nl
        valid = (nl >= self.min_samples_leaf) & (nr >= self.min_samples_leaf)
        if not valid.any():
            return None
        ids = np.flatnonzero(valid)
        scores = np.asarray(frequencies)[ids]
        if self.eta:
            left_square, right_square = np.zeros(len(ids)), np.zeros(len(ids))
            for label in np.unique(y):
                counts = masks[y == label][:, ids].sum(axis=0)
                left_square += counts**2
                right_square += (np.sum(y == label) - counts) ** 2
            loss = (len(y) - left_square / nl[ids] - right_square / nr[ids]) / (
                2 * len(y)
            )
            scores = scores * np.exp(-self.eta * loss)
        best = choose_min(-scores, self.rng_ if self.random_ties else None)
        return candidates[ids[best]]
