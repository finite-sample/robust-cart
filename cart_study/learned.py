"""Learn partition groups and mix ordinary with gap-coarsened CART proposals."""

import argparse
import hashlib
import json
import os
import platform
import tempfile
from collections import Counter
from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from cart_study.ablation import (
    CONTEXTS,
    PartitionTree,
    accuracy,
    background,
    interval,
    proposal_splits,
    require_equal,
    require_same_tree,
    rng_for,
    structure,
    verify_builder,
)
from cart_study.baselines import ALOOFTree, NodeVoteTree
from cart_study.trees import leaf_count

PRIMARY = [
    "grouping_raw",
    "grouping_mixed",
    "proposals_peak",
    "proposals_grouped",
    "interaction",
    "combined_vs_cart",
    "combined_vs_gap_cart",
    "combined_vs_tuned_cart",
    "combined_vs_aloof",
    "combined_vs_literal",
]
ARMS = [
    "raw_peak",
    "raw_grouped",
    "mixed_peak",
    "mixed_grouped",
    "literal",
    "cart",
    "gap_cart",
    "tuned_cart",
    "aloof",
    "node_vote",
]


class GapMap:
    """Flag a gap only when it is wider than both observed within-side ranges."""

    def __init__(self, min_leaf=10):
        self.min_leaf = min_leaf

    def fit(self, X):
        self.thresholds = {}
        for feature, values in enumerate(np.sort(X, axis=0).T):
            cuts = np.arange(self.min_leaf, len(values) - self.min_leaf + 1)
            if not len(cuts):
                continue
            gaps = values[cuts] - values[cuts - 1]
            k = int(cuts[np.argmax(gaps)])
            gap = values[k] - values[k - 1]
            if gap > max(values[k - 1] - values[0], values[-1] - values[k]):
                self.thresholds[feature] = midpoint(values[k - 1], values[k])
        return self

    def transform(self, X):
        result = np.array(X, dtype=float, copy=True)
        for feature, threshold in self.thresholds.items():
            result[:, feature] = X[:, feature] > threshold
        return result


def midpoint(lower, upper):
    value = float(lower / 2 + upper / 2)
    return value if value < upper else float(lower)


def partition_groups(X, pool):
    """Learn groups from training covariates; no labels or population metadata."""
    ordered = np.sort(X, axis=0)
    exact = Counter(s for splits in pool for s in splits)
    candidates, peak, union = {}, Counter(), Counter()
    keys = {}
    for (feature, threshold), count in exact.items():
        rank = int(np.searchsorted(ordered[:, feature], threshold, side="right"))
        if rank == 0 or rank == len(X):
            continue
        key = feature, rank
        keys[feature, threshold] = key
        candidates[key] = (
            feature,
            midpoint(ordered[rank - 1, feature], ordered[rank, feature]),
        )
        peak[key] = max(peak[key], count)
    for splits in pool:
        union.update({keys[s] for s in splits if s in keys})
    if any(not 1 <= peak[k] <= union[k] <= len(pool) for k in candidates):
        raise AssertionError("Invalid per-tree support")
    return candidates, dict(peak), dict(union)


def priorities(candidates, key):
    return {s: hashlib.sha256(repr((*key, s)).encode()).hexdigest() for s in candidates}


def cart_fit(X, y, options, seed, order, mapping=None):
    transformed = X if mapping is None else mapping.transform(X)
    tree = DecisionTreeClassifier(**options, random_state=int(seed)).fit(
        transformed[:, order], y
    )
    features = tree.tree_.feature
    features[features >= 0] = order[features[features >= 0]]
    if mapping is not None:
        for node, feature in enumerate(features):
            if feature in mapping.thresholds:
                if tree.tree_.threshold[node] != 0.5:
                    raise AssertionError("Coarsened feature had a nonbinary split")
                tree.tree_.threshold[node] = mapping.thresholds[feature]
    return tree


def fit_factorial(X, y, samples, seeds, order, depth, mapping, key):
    options = dict(max_depth=depth, min_samples_leaf=10)
    raw = [
        cart_fit(X[ids], y[ids], options, seed, order)
        for ids, seed in zip(samples, seeds)
    ]
    mixed = raw[: len(raw) // 2] + [
        cart_fit(X[ids], y[ids], options, seed, order, mapping)
        for ids, seed in zip(samples[len(raw) // 2 :], seeds[len(raw) // 2 :])
    ]
    models, components = {}, {}
    for name, proposals in [("raw", raw), ("mixed", mixed)]:
        pool = [proposal_splits(tree, depth) for tree in proposals]
        candidates, peak, union = partition_groups(X, pool)
        tie = priorities(candidates, key)
        for rule, scores in [("peak", peak), ("grouped", union)]:
            arm = f"{name}_{rule}"
            models[arm] = PartitionTree(candidates, scores, tie, **options).fit(X, y)
            components[arm] = (candidates, scores, tie)
    exact = Counter(s for tree in raw for s in proposal_splits(tree, depth))
    candidates = {(f, float(t).hex()): (f, t) for f, t in exact}
    models["literal"] = PartitionTree(
        candidates,
        {(f, float(t).hex()): score for (f, t), score in exact.items()},
        priorities(candidates, key),
        **options,
    ).fit(X, y)
    return models, components, raw


def tuned_carts(X, y, order, seeds, depths):
    """Shared three-fold tuning; each comparison respects its maximum depth."""
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=int(seeds[-1]))
    folds = list(splitter.split(X, y))
    grid = [
        dict(max_depth=d, min_samples_leaf=m, ccp_alpha=a)
        for d, m, a in product(depths, [5, 10, 20], [0.0, 0.01, 0.03])
    ]
    errors = []
    for options in grid:
        mistakes = 0
        for fold, (train, validation) in enumerate(folds):
            model = cart_fit(X[train], y[train], options, seeds[fold], order)
            mistakes += np.count_nonzero(model.predict(X[validation]) != y[validation])
        errors.append(mistakes)
    result = {}
    for depth in depths:
        eligible = [i for i, option in enumerate(grid) if option["max_depth"] <= depth]
        best = min(
            eligible,
            key=lambda i: (
                errors[i],
                grid[i]["max_depth"],
                -grid[i]["ccp_alpha"],
                -grid[i]["min_samples_leaf"],
            ),
        )
        result[depth] = cart_fit(X, y, grid[best], seeds[-1], order)
    return result


def conditions():
    result = []
    for context in CONTEXTS:
        for n in [200, 800]:
            result.append(dict(context=context, n=n, eta=0.4, delta=0.25))
    for context in ["gaussian", "xor", "majority"]:
        result.append(dict(context=context, n=200, eta=0.1, delta=0.25))
    for delta in [0.0, 0.1, 0.4, 0.6]:
        result.append(dict(context="gaussian", n=200, eta=0.4, delta=delta))
    for context in ["continuous", "within_cluster", "imbalanced", "contaminated"]:
        for eta in [0.1, 0.4]:
            result.append(dict(context=context, n=200, eta=eta, delta=0.25))
    result.append(dict(context="gaussian", n=200, eta=0.5, delta=0.25))
    return result


def observations(spec, size, seed, condition, replicate, offset):
    def rng(stream):
        return rng_for(seed, condition, replicate, offset + stream)

    context = spec["context"]
    signals = {"xor": 2, "majority": 3}.get(context, 1)
    probability = 0.2 if context == "imbalanced" else 0.5
    bits = (rng(0).uniform(size=(size, signals)) < probability).astype(int)
    if context in ["breast_cancer", "wine", "digits"]:
        source = background(context)
        noise = source[rng(1).integers(len(source), size=size)]
    else:
        noise = rng(1).normal(size=(size, 20 - signals))
        if context == "correlated":
            noise = np.sqrt(0.8) * rng(5).normal(size=(size, 1)) + np.sqrt(0.2) * noise
        extra = {"binary": 19, "mixed": 9}.get(context, 0)
        if extra:
            bits = np.column_stack((bits, rng(2).integers(2, size=(size, extra))))
            noise = noise[:, extra:]
    jitter = rng(3).uniform(-1, 1, size=bits.shape)
    X = np.column_stack((bits + spec["delta"] * jitter, noise))
    truth = bits[:, 0].copy()
    if context == "xor":
        truth = bits[:, 0] ^ bits[:, 1]
    elif context == "majority":
        truth = (bits[:, :3].sum(axis=1) >= 2).astype(int)
    elif context == "within_cluster":
        truth = (jitter[:, 0] > 0).astype(int)
    elif context == "continuous":
        X[:, 0] = jitter[:, 0]
        truth = (X[:, 0] > 0).astype(int)
    elif context == "contaminated":
        X[:, 0] += 10 * (rng(6).uniform(size=size) < 0.01)
    # CART uses float32 internally; use identical covariate values in every arm.
    return X.astype(np.float32).astype(float), truth, rng(4).uniform(size=size)


def validate_training_geometry(
    X, y, Xt, mapping, models, components, raw, options, seed, checks
):
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(X.shape[1])
    rows = rng.permutation(len(X))
    reordered = GapMap(options["min_samples_leaf"]).fit(X[rows])
    require_equal(mapping.thresholds, reordered.thresholds, checks, "gap_row_order")
    permuted = GapMap(options["min_samples_leaf"]).fit(X[:, permutation])
    require_equal(
        mapping.thresholds,
        {int(permutation[f]): v for f, v in permuted.thresholds.items()},
        checks,
        "gap_column_order",
    )
    scaled = GapMap(options["min_samples_leaf"]).fit(2 * X)
    require_equal(
        mapping.thresholds,
        {f: v / 2 for f, v in scaled.thresholds.items()},
        checks,
        "gap_scale",
    )
    for arm, (candidates, scores, tie) in components.items():
        verify_builder(
            models[arm], X, y, Xt, candidates, scores, tie, options, permutation, checks
        )
        shuffled = PartitionTree(candidates, scores, tie, **options).fit(
            X[rows], y[rows]
        )
        require_same_tree(structure(models[arm]), structure(shuffled), checks)
    if not mapping.thresholds:
        for rule in ["peak", "grouped"]:
            require_same_tree(
                structure(models[f"raw_{rule}"]),
                structure(models[f"mixed_{rule}"]),
                checks,
            )
        checks["no_coarsening_identity"] += 1
    pool = [proposal_splits(tree, options["max_depth"]) for tree in raw]
    candidates, peak, union = partition_groups(X, pool)
    require_equal(
        (candidates, peak, union),
        partition_groups(X[rows], pool),
        checks,
        "group_row_order",
    )
    ordered = np.sort(X, axis=0)
    aliased = []
    for b, splits in enumerate(pool):
        moved = set()
        for feature, threshold in splits:
            rank = int(np.searchsorted(ordered[:, feature], threshold, side="right"))
            if 0 < rank < len(X):
                lower, upper = ordered[rank - 1, feature], ordered[rank, feature]
                weight = (b + 1) / (len(pool) + 1)
                threshold = min(
                    lower + weight * (upper - lower), np.nextafter(upper, lower)
                )
            moved.add((feature, float(threshold)))
        aliased.append(moved)
    alias_candidates, _, alias_union = partition_groups(X, aliased)
    require_equal(candidates, alias_candidates, checks, "alias_candidates")
    require_equal(union, alias_union, checks, "alias_grouped_votes")
    alias_model = PartitionTree(
        alias_candidates, alias_union, components["raw_grouped"][2], **options
    ).fit(X, y)
    require_same_tree(structure(models["raw_grouped"]), structure(alias_model), checks)
    disagreement = []
    for original in raw:
        canonical = deepcopy(original)
        for node, feature in enumerate(canonical.tree_.feature):
            if feature < 0:
                continue
            threshold = canonical.tree_.threshold[node]
            rank = int(np.searchsorted(ordered[:, feature], threshold, side="right"))
            if 0 < rank < len(X):
                canonical.tree_.threshold[node] = midpoint(
                    ordered[rank - 1, feature], ordered[rank, feature]
                )
        require_equal(
            original.predict(X),
            canonical.predict(X),
            checks,
            "canonical_training_predictions",
        )
        disagreement.append(
            float(np.mean(original.predict(Xt) != canonical.predict(Xt)))
        )
    return float(np.mean(disagreement))


def run_unit(config, condition, replicate):
    spec = config["conditions"][condition]
    seed, n, eta = config["seed"], spec["n"], spec["eta"]
    X, truth, flips = observations(spec, n, seed, condition, replicate, 0)
    y = truth ^ (flips < eta)
    Xt, truth_test, flips_test = observations(
        spec, config["test_size"], seed, condition, replicate, 100
    )
    yt = truth_test ^ (flips_test < eta)
    mapping = GapMap(config["min_leaf"]).fit(X)
    samples = rng_for(seed, condition, replicate, 200).integers(
        n, size=(config["pool_size"], n)
    )
    seeds = rng_for(seed, condition, replicate, 201).integers(
        2**31, size=config["pool_size"] + 1
    )
    order = rng_for(seed, condition, replicate, 202).permutation(X.shape[1])
    checks, outcomes, contrasts = Counter(), [], []
    with threadpool_limits(limits=1):
        tuned = tuned_carts(X, y, order, seeds, config["depths"])
        for depth in config["depths"]:
            options = dict(max_depth=depth, min_samples_leaf=config["min_leaf"])
            models, components, raw = fit_factorial(
                X,
                y,
                samples,
                seeds[:-1],
                order,
                depth,
                mapping,
                (seed, condition, replicate, depth),
            )
            drift = validate_training_geometry(
                X,
                y,
                Xt,
                mapping,
                models,
                components,
                raw,
                options,
                int(seeds[-1]),
                checks,
            )
            models["cart"] = cart_fit(X, y, options, seeds[-1], order)
            models["gap_cart"] = cart_fit(X, y, options, seeds[-1], order, mapping)
            models["tuned_cart"] = tuned[depth]
            for arm, cls, kwargs in [
                ("aloof", ALOOFTree, {}),
                ("node_vote", NodeVoteTree, dict(n_subsamples=20)),
            ]:
                model = cls(**options, random_state=int(seeds[-1]), **kwargs).fit(
                    X[:, order], y
                )
                stack = [model.tree_]
                while stack:
                    node = stack.pop()
                    if node.feature >= 0:
                        node.feature = int(order[node.feature])
                        stack.extend([node.left, node.right])
                models[arm] = model
            predictions, measured = {}, {}
            for arm in ARMS:
                pred = models[arm].predict(Xt)
                predictions[arm] = pred
                measured[arm] = accuracy(pred, truth_test, eta)
                if eta == 0.5:
                    require_equal(measured[arm], 0.5, checks, "null_risk")
                outcomes.append(
                    dict(
                        depth=depth,
                        arm=arm,
                        accuracy=measured[arm],
                        observed_accuracy=float(np.mean(pred == yt)),
                        leaves=leaf_count(models[arm]),
                        candidates=(
                            len(components[arm][0]) if arm in components else None
                        ),
                    )
                )
            pairs = dict(
                grouping_raw=("raw_grouped", "raw_peak"),
                grouping_mixed=("mixed_grouped", "mixed_peak"),
                proposals_peak=("mixed_peak", "raw_peak"),
                proposals_grouped=("mixed_grouped", "raw_grouped"),
            )
            pairs.update(
                {
                    f"combined_vs_{arm}": ("mixed_grouped", arm)
                    for arm in [
                        "cart",
                        "gap_cart",
                        "tuned_cart",
                        "aloof",
                        "literal",
                        "node_vote",
                    ]
                }
            )
            effects = {}
            for name, (a, b) in pairs.items():
                helped = float(
                    np.mean(
                        (predictions[a] == truth_test) & (predictions[b] != truth_test)
                    )
                )
                harmed = float(
                    np.mean(
                        (predictions[a] != truth_test) & (predictions[b] == truth_test)
                    )
                )
                effect = (1 - 2 * eta) * (helped - harmed)
                effects[name] = effect
                contrasts.append(
                    dict(
                        depth=depth,
                        contrast=name,
                        effect=effect,
                        helped=helped,
                        harmed=harmed,
                    )
                )
            contrasts.append(
                dict(
                    depth=depth,
                    contrast="interaction",
                    effect=effects["grouping_mixed"] - effects["grouping_raw"],
                    helped=None,
                    harmed=None,
                )
            )
            outcomes.append(
                dict(
                    depth=depth,
                    arm="diagnostic",
                    accuracy=None,
                    observed_accuracy=None,
                    leaves=None,
                    candidates=None,
                    canonical_test_disagreement=drift,
                )
            )
    digest = hashlib.sha256()
    for array in [X, y, Xt, truth_test, samples, seeds, order]:
        digest.update(array.tobytes())
    return dict(
        condition=condition,
        replicate=replicate,
        spec=spec,
        input_hash=digest.hexdigest(),
        flagged_features=sorted(mapping.thresholds),
        outcomes=outcomes,
        contrasts=contrasts,
        checks=dict(checks),
    )


def summarize(output, config):
    units = [
        json.loads((output / "units" / f"{c:02}_{r:04}.json").read_text())
        for c in range(len(config["conditions"]))
        for r in range(config["repetitions"])
    ]
    outcomes = pd.DataFrame(
        dict(condition=u["condition"], replicate=u["replicate"], **u["spec"], **row)
        for u in units
        for row in u["outcomes"]
    )
    contrasts = pd.DataFrame(
        dict(condition=u["condition"], replicate=u["replicate"], **u["spec"], **row)
        for u in units
        for row in u["contrasts"]
    )
    expected = len(units) * len(config["depths"]) * (len(ARMS) + 1)
    if (
        len(outcomes) != expected
        or outcomes.duplicated(["condition", "replicate", "depth", "arm"]).any()
    ):
        raise AssertionError("Incomplete or duplicated experiment cells")
    keys = ["condition", "context", "n", "eta", "delta", "depth"]
    family = len(config["conditions"]) * len(config["depths"]) * len(PRIMARY)
    estimates = []
    for names, frame in contrasts.groupby(keys + ["contrast"]):
        if len(frame) != config["repetitions"]:
            raise AssertionError("Incomplete paired contrasts")
        row = dict(zip(keys + ["contrast"], names))
        row.update(interval(100 * frame.effect))
        row["repetitions"] = len(frame)
        if row["contrast"] in PRIMARY:
            simultaneous = interval(100 * frame.effect, family)
            row.update(
                simultaneous_lower=simultaneous["lower"],
                simultaneous_upper=simultaneous["upper"],
            )
        estimates.append(row)
    pd.DataFrame(estimates).to_csv(output / "contrasts.csv", index=False)
    outcomes.groupby(keys + ["arm"]).agg(
        accuracy=("accuracy", "mean"),
        mcse=("accuracy", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        observed_accuracy=("observed_accuracy", "mean"),
        leaves=("leaves", "mean"),
        candidates=("candidates", "mean"),
        canonical_test_disagreement=("canonical_test_disagreement", "mean"),
    ).reset_index().to_csv(output / "summary.csv", index=False)
    flags = pd.DataFrame(
        dict(
            condition=u["condition"],
            **u["spec"],
            flagged_count=len(u["flagged_features"]),
            first_feature_flagged=int(0 in u["flagged_features"]),
        )
        for u in units
    )
    flags.groupby(keys[:-1]).mean().reset_index().to_csv(
        output / "gap_detection.csv", index=False
    )
    checks = sum((Counter(u["checks"]) for u in units), Counter())
    (output / "invariances.json").write_text(json.dumps(dict(checks), indent=2) + "\n")
    plot(pd.DataFrame(estimates), output)
    print(
        f"Completed {len(units)} paired replicates; {family} primary contrasts.",
        flush=True,
    )


def plot(frame, output):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cart-mpl"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = frame[(frame.n == 200) & (frame.eta == 0.4) & (frame.delta == 0.25)]
    contexts = CONTEXTS + ["continuous", "within_cluster", "imbalanced", "contaminated"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5), sharey=True)
    for ax, contrast, title in zip(
        axes,
        ["combined_vs_tuned_cart", "combined_vs_gap_cart"],
        ["Compared with tuned CART", "Compared with gap-coarsened CART"],
    ):
        for depth, shift, color in [(2, -0.12, "#23678a"), (5, 0.12, "#a04b30")]:
            rows = selected[(selected.contrast == contrast) & (selected.depth == depth)]
            rows = rows.set_index("context").reindex(contexts).dropna(subset=["mean"])
            positions = np.array([contexts.index(c) for c in rows.index]) + shift
            ax.errorbar(
                rows["mean"],
                positions,
                xerr=[
                    rows["mean"] - rows.simultaneous_lower,
                    rows.simultaneous_upper - rows["mean"],
                ],
                fmt="o",
                color=color,
                capsize=2,
                label=f"Depth {depth}",
            )
        ax.axvline(0, color="#777777", linewidth=0.8)
        ax.set(title=title, xlabel="Accuracy difference (percentage points)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(
        np.arange(len(contexts)), [c.replace("_", " ").capitalize() for c in contexts]
    )
    axes[0].invert_yaxis()
    axes[1].legend(frameon=False)
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"accuracy.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def compute_unit(output, config, condition, replicate):
    path = output / "units" / f"{condition:02}_{replicate:04}.json"
    if path.exists():
        return
    unit = run_unit(config, condition, replicate)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(unit, separators=(",", ":"), allow_nan=False) + "\n"
    )
    temporary.replace(path)


def main():
    from cart_study.learned import compute_unit

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/learned"))
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.repetitions < 2 or args.jobs < 1 or args.seed < 0:
        parser.error("Need repetitions >= 2, jobs >= 1, seed >= 0")
    if args.summarize:
        summarize(args.output, json.loads((args.output / "config.json").read_text()))
        return
    config = dict(
        seed=args.seed,
        repetitions=2 if args.smoke else args.repetitions,
        test_size=200 if args.smoke else 5000,
        conditions=conditions(),
        pool_size=20,
        min_leaf=10,
        depths=[2, 5],
        primary=PRIMARY,
        proposal="20 raw versus 10 raw + 10 gap-coarsened, paired bootstrap rows/seeds",
        grouping=(
            "Identical partitions on full training X; "
            "common midpoints; peak versus per-tree union"
        ),
        geometry=(
            "Gap map and partition groups learned on all training covariates; "
            "shared by bootstrap proposals; no population gaps or evaluation data"
        ),
        gap="Largest admissible adjacent gap exceeds both within-side observed ranges",
        inference="Replicate-paired t intervals; Bonferroni over all primary contrasts",
        tuning=(
            "Three-fold CART: depths2/5 within cap, "
            "leaf5/10/20, alpha0/.01/.03; labels only inside training"
        ),
        prior_information=(
            "Oracle ablation informed design and width .25; "
            "no outcomes used to tune gap rule"
        ),
        scope=(
            "All constructed bits jittered; real covariates "
            "independent nuisance; no real outcomes used"
        ),
        missing="Failed fit or invariant aborts; no dropped units; exact-config resume",
        versions=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            sklearn=sklearn.__version__,
        ),
        source_hashes={
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ["learned.py", "ablation.py", "trees.py", "baselines.py"]
        },
    )
    serialized = json.dumps(config, indent=2) + "\n"
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "config.json"
    if path.exists() and path.read_text() != serialized:
        raise ValueError(
            "Existing run has a different source or design; choose another output"
        )
    path.write_text(serialized)
    (args.output / "units").mkdir(exist_ok=True)
    Parallel(n_jobs=args.jobs, verbose=10)(
        delayed(compute_unit)(args.output, config, c, r)
        for c in range(len(config["conditions"]))
        for r in range(config["repetitions"])
    )
    summarize(args.output, config)


if __name__ == "__main__":
    main()
