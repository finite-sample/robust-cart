"""Paired interventions on threshold identity and predictor representation."""

import argparse
import hashlib
import json
import os
import platform
import tempfile
from collections import Counter, defaultdict
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from joblib import Parallel, delayed
from scipy.stats import t
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from cart_study.trees import ExperimentalTree, FastPrediction, leaf_count, split_set

CONTEXTS = [
    "gaussian",
    "binary",
    "mixed",
    "correlated",
    "xor",
    "majority",
    "breast_cancer",
    "wine",
    "digits",
]
REGIMES = [(200, 0.4), (200, 0.1), (800, 0.4)]
ARMS = ["exact", "grouped", "fragmented", "cart"]
PRIMARY = [
    "fragmentation_binary",
    "fragmentation_jittered",
    "aggregation_binary",
    "aggregation_jittered",
    "interaction",
]


def rng_for(seed, condition, replicate, stream):
    return np.random.default_rng(
        np.random.SeedSequence([seed, condition, replicate, stream])
    )


@lru_cache
def background(context):
    return {
        "breast_cancer": load_breast_cancer,
        "wine": load_wine,
        "digits": load_digits,
    }[context]().data


def observations(context, size, seed, condition, replicate, offset):
    """Independent streams for latent bits, covariates, jitter, and label flips."""

    def rng(stream):
        return rng_for(seed, condition, replicate, offset + stream)

    signals = {"xor": 2, "majority": 3}.get(context, 1)
    bits = rng(0).integers(2, size=(size, signals))
    truth = bits[:, 0].copy()
    if context == "xor":
        truth = bits[:, 0] ^ bits[:, 1]
    elif context == "majority":
        truth = (bits.sum(axis=1) >= 2).astype(int)
    if context in {"breast_cancer", "wine", "digits"}:
        source = background(context)
        noise = source[rng(1).integers(len(source), size=size)]
    else:
        noise = rng(1).normal(size=(size, 20 - signals))
        if context == "correlated":
            noise = np.sqrt(0.8) * rng(5).normal(size=(size, 1)) + np.sqrt(0.2) * noise
        binary_count = {"binary": 19, "mixed": 9}.get(context, 0)
        if binary_count:
            bits = np.column_stack(
                (bits, rng(2).integers(2, size=(size, binary_count)))
            )
            noise = noise[:, binary_count:]
    binary = np.column_stack((bits, noise)).astype(float)
    jittered = binary.copy()
    jittered[:, : bits.shape[1]] += 0.25 * rng(3).uniform(-1, 1, size=bits.shape)
    if not np.array_equal(jittered[:, : bits.shape[1]] > 0.5, bits):
        raise AssertionError("Representation changed latent bits")
    return binary, jittered, truth, rng(4).uniform(size=size), bits.shape[1], signals


def proposal_splits(model, depth):
    result = set()
    stack = [(0, 0)]
    while stack:
        node, level = stack.pop()
        if level == depth or model.tree_.feature[node] < 0:
            continue
        result.add((int(model.tree_.feature[node]), float(model.tree_.threshold[node])))
        stack.extend(
            [
                (model.tree_.children_left[node], level + 1),
                (model.tree_.children_right[node], level + 1),
            ]
        )
    return result


def group_key(split, bit_count, delta):
    feature, threshold = split
    # Only the known support gap is collapsed, never an empirical empty gap.
    if feature < bit_count and delta <= threshold < 1 - delta:
        return feature, "gap"
    return feature, float(threshold).hex()


def aliased_pool(pool, bit_count, delta, fragmented):
    result = []
    for b, splits in enumerate(pool):
        alias = delta + (1 - 2 * delta) * (b + 1) / (len(pool) + 1)
        result.append(
            {
                (
                    (f, alias if fragmented else 0.5)
                    if group_key((f, v), bit_count, delta)[1] == "gap"
                    else (f, v)
                )
                for f, v in splits
            }
        )
    return result


def support(pool, bit_count, delta):
    exact = Counter(split for splits in pool for split in splits)
    union = Counter(
        key
        for splits in pool
        for key in {group_key(s, bit_count, delta) for s in splits}
    )
    peak = defaultdict(int)
    for split, count in exact.items():
        key = group_key(split, bit_count, delta)
        peak[key] = max(peak[key], count)
    if any(not 1 <= peak[k] <= union[k] <= len(pool) for k in union):
        raise AssertionError("Support must count each tree at most once")
    return dict(peak), dict(union)


class PartitionTree(FastPrediction, ExperimentalTree):
    """Common candidate partitions, fixed priorities, and one score per partition."""

    def __init__(self, candidates, scores, priorities, **kwargs):
        super().__init__(**kwargs)
        self.ranked = sorted(candidates, key=lambda k: (-scores[k], priorities[k], k))
        self.candidates = candidates

    def _choose(self, X, y):
        for key in self.ranked:
            feature, threshold = self.candidates[key]
            count = np.count_nonzero(X[:, feature] <= threshold)
            if self.min_samples_leaf <= count <= len(y) - self.min_samples_leaf:
                return feature, threshold
        return None


def require_equal(left, right, checks, name):
    if not np.array_equal(left, right):
        raise AssertionError(name)
    checks[name] += 1


def structure(model, feature_map=None, scale=1):
    def visit(node):
        if node.feature < 0:
            return ("leaf", int(node.prediction), node.count)
        feature = node.feature if feature_map is None else feature_map[node.feature]
        return (
            int(feature),
            node.threshold / scale,
            int(node.prediction),
            node.count,
            visit(node.left),
            visit(node.right),
        )

    return visit(model.tree_)


def require_same_tree(expected, actual, checks):
    if expected != actual:
        raise AssertionError("Fixed-pool tree structure changed")
    checks["tree_structure"] += 1


def freeze_candidates(pool, bit_count, delta):
    candidates = {}
    for split in sorted(set().union(*pool)):
        candidates.setdefault(group_key(split, bit_count, delta), split)
    return candidates


def verify_aliases(models, X, Xt, bit_count, delta, checks):
    """Replay actual proposal trees after moving every eligible threshold."""
    joined = np.concatenate((X, Xt))
    for b, original in enumerate(models):
        expected = original.predict(joined)
        for fragmented in (False, True):
            changed = deepcopy(original)
            alias = delta + (1 - 2 * delta) * (b + 1) / (len(models) + 1)
            moved = 0
            for node, feature in enumerate(changed.tree_.feature):
                if feature < 0:
                    continue
                threshold = changed.tree_.threshold[node]
                if group_key((feature, threshold), bit_count, delta)[1] == "gap":
                    changed.tree_.threshold[node] = alias if fragmented else 0.5
                    moved += 1
            checks["proposal_thresholds_moved"] += moved
            require_equal(
                expected,
                changed.predict(joined),
                checks,
                "proposal_predictions_preserved",
            )


def verify_builder(
    model, X, y, Xt, candidates, scores, priorities, options, permutation, checks
):
    expected = model.predict(Xt)

    def fit(c):
        return PartitionTree(c, scores, priorities, **options)

    reordered = fit(dict(reversed(list(candidates.items())))).fit(X, y)
    require_same_tree(structure(model), structure(reordered), checks)
    require_equal(expected, reordered.predict(Xt), checks, "candidate_order")
    inverse = np.argsort(permutation)
    remapped = {key: (int(inverse[f]), v) for key, (f, v) in candidates.items()}
    permuted = fit(remapped).fit(X[:, permutation], y)
    require_same_tree(structure(model), structure(permuted, permutation), checks)
    require_equal(
        expected, permuted.predict(Xt[:, permutation]), checks, "column_order"
    )
    scaled = fit({key: (f, 2 * v) for key, (f, v) in candidates.items()}).fit(2 * X, y)
    require_same_tree(structure(model), structure(scaled, scale=2), checks)
    require_equal(expected, scaled.predict(2 * Xt), checks, "scale_by_two")
    unused = sorted(set(range(X.shape[1])) - {f for f, _ in split_set(model)})
    if unused:
        altered = Xt.copy()
        altered[:, unused] = 1e6
        require_equal(expected, model.predict(altered), checks, "unused_features")
    else:
        checks["unused_features_ineligible"] += 1


def accuracy(prediction, truth, eta):
    """Integrate out independent evaluation-label flips exactly."""
    return float(1 - eta - (1 - 2 * eta) * np.mean(prediction != truth))


def run_unit(config, condition, replicate):
    context, n, eta = config["conditions"][condition]
    seed = config["seed"]
    checks = Counter()
    X0, X1, truth, flips, bit_count, signals = observations(
        context, n, seed, condition, replicate, 0
    )
    y = truth ^ (flips < eta)
    Xt0, Xt1, truth_test, test_flips, _, _ = observations(
        context, config["test_size"], seed, condition, replicate, 100
    )
    yt = truth_test ^ (test_flips < eta)
    checks["latent_bits_preserved"] += 2
    rng = rng_for(seed, condition, replicate, 200)
    samples = rng.integers(n, size=(config["pool_size"], n))
    tree_seeds = rng_for(seed, condition, replicate, 201).integers(
        2**31, size=config["pool_size"] + 1
    )
    # Random positions are shared by both representations; bit IDs stay attached.
    order = rng_for(seed, condition, replicate, 202).permutation(X0.shape[1])
    inverse = np.argsort(order)
    extra_order = rng_for(seed, condition, replicate, 203).permutation(X0.shape[1])
    rows, contrasts, predictions = [], [], {}
    with threadpool_limits(limits=1):
        for representation, X, Xt, delta in [
            ("binary", X0, Xt0, 0.0),
            ("jittered", X1, Xt1, 0.25),
        ]:
            pool_models = []
            for sample, tree_seed in zip(samples, tree_seeds):
                tree = DecisionTreeClassifier(
                    max_depth=max(config["depths"]),
                    min_samples_leaf=config["min_leaf"],
                    random_state=int(tree_seed),
                ).fit(X[sample][:, order], y[sample])
                # Restore semantic feature IDs for frozen-pool interventions.
                features = tree.tree_.feature
                features[features >= 0] = order[features[features >= 0]]
                pool_models.append(tree)
            verify_aliases(pool_models, X, Xt, bit_count, delta, checks)
            for depth in config["depths"]:
                options = dict(max_depth=depth, min_samples_leaf=config["min_leaf"])
                pool = [proposal_splits(tree, depth) for tree in pool_models]
                fingerprint = repr(pool)
                candidates = freeze_candidates(pool, bit_count, delta)
                exact, grouped = support(pool, bit_count, delta)
                canonical_exact, canonical_grouped = support(
                    aliased_pool(pool, bit_count, delta, False), bit_count, delta
                )
                fragmented_exact, fragmented_grouped = support(
                    aliased_pool(pool, bit_count, delta, True), bit_count, delta
                )
                require_equal(grouped, canonical_exact, checks, "canonical_support")
                require_equal(grouped, canonical_grouped, checks, "canonical_union")
                require_equal(grouped, fragmented_grouped, checks, "fragmented_union")
                priorities = {
                    key: hashlib.sha256(
                        repr((seed, condition, replicate, depth, key)).encode()
                    ).hexdigest()
                    for key in candidates
                }
                score_arms = dict(
                    exact=exact, grouped=grouped, fragmented=fragmented_exact
                )
                models = {
                    arm: PartitionTree(candidates, score, priorities, **options).fit(
                        X, y
                    )
                    for arm, score in score_arms.items()
                }
                group_predictions = models["grouped"].predict(Xt)
                for score in [canonical_exact, canonical_grouped, fragmented_grouped]:
                    check_model = PartitionTree(
                        candidates, score, priorities, **options
                    ).fit(X, y)
                    require_same_tree(
                        structure(models["grouped"]), structure(check_model), checks
                    )
                    require_equal(
                        group_predictions,
                        check_model.predict(Xt),
                        checks,
                        "grouped_predictions",
                    )
                if exact == grouped:
                    require_equal(
                        group_predictions,
                        models["exact"].predict(Xt),
                        checks,
                        "no_fragmentation",
                    )
                for arm, score in score_arms.items():
                    verify_builder(
                        models[arm],
                        X,
                        y,
                        Xt,
                        candidates,
                        score,
                        priorities,
                        options,
                        extra_order,
                        checks,
                    )
                cart = DecisionTreeClassifier(
                    **options, random_state=int(tree_seeds[-1])
                )
                cart.fit(X[:, order], y)
                cart_predictions = cart.predict(Xt[:, order])
                permuted_cart = DecisionTreeClassifier(
                    **options, random_state=int(tree_seeds[-1])
                ).fit(X[:, extra_order], y)
                permutation_predictions = permuted_cart.predict(Xt[:, extra_order])
                for arm in ARMS:
                    pred = (
                        cart_predictions if arm == "cart" else models[arm].predict(Xt)
                    )
                    predictions[representation, depth, arm] = pred
                    measured = accuracy(pred, truth_test, eta)
                    if eta == 0.5:
                        require_equal(measured, 0.5, checks, "outcome_null")
                    informative = set(range(signals))
                    root = (
                        int(cart.tree_.feature[0])
                        if arm == "cart"
                        else models[arm].tree_.feature
                    )
                    if arm == "cart" and root >= 0:
                        root = int(order[root])
                    rows.append(
                        dict(
                            context=context,
                            n=n,
                            eta=eta,
                            replicate=replicate,
                            representation=representation,
                            depth=depth,
                            arm=arm,
                            accuracy=measured,
                            observed_accuracy=float(np.mean(pred == yt)),
                            leaves=leaf_count(cart if arm == "cart" else models[arm]),
                            signal_root=int(root in informative),
                            candidates=len(candidates),
                            eligible_groups=sum(k[1] == "gap" for k in candidates),
                            natural_fragmented_groups=sum(
                                exact[k] < grouped[k] for k in candidates
                            ),
                            signal_peak=max(
                                (v for (f, _), v in exact.items() if f < signals),
                                default=0,
                            )
                            / config["pool_size"],
                            signal_union=max(
                                (v for (f, _), v in grouped.items() if f < signals),
                                default=0,
                            )
                            / config["pool_size"],
                            column_refit_disagreement=(
                                float(
                                    np.mean(cart_predictions != permutation_predictions)
                                )
                                if arm == "cart"
                                else None
                            ),
                            column_refit_accuracy_change=(
                                accuracy(permutation_predictions, truth_test, eta)
                                - measured
                                if arm == "cart"
                                else None
                            ),
                        )
                    )
                require_equal(fingerprint, repr(pool), checks, "pool_unchanged")
        for depth in config["depths"]:
            effects = {}
            for representation in ["binary", "jittered"]:
                pred = {arm: predictions[representation, depth, arm] for arm in ARMS}
                for name, a, b in [
                    ("fragmentation", "grouped", "fragmented"),
                    ("aggregation", "grouped", "exact"),
                    ("grouped_vs_cart", "grouped", "cart"),
                    ("exact_vs_cart", "exact", "cart"),
                ]:
                    helped = np.mean((pred[a] == truth_test) & (pred[b] != truth_test))
                    harmed = np.mean((pred[a] != truth_test) & (pred[b] == truth_test))
                    effect = float((1 - 2 * eta) * (helped - harmed))
                    effects[f"{name}_{representation}"] = effect
                    contrasts.append(
                        dict(
                            context=context,
                            n=n,
                            eta=eta,
                            replicate=replicate,
                            depth=depth,
                            contrast=f"{name}_{representation}",
                            effect=effect,
                            helped=float(helped),
                            harmed=float(harmed),
                        )
                    )
            contrasts.append(
                dict(
                    context=context,
                    n=n,
                    eta=eta,
                    replicate=replicate,
                    depth=depth,
                    contrast="interaction",
                    effect=effects["aggregation_jittered"]
                    - effects["aggregation_binary"],
                    helped=None,
                    harmed=None,
                )
            )
    # Identify all shared random inputs; the seed keys above reproduce each array.
    digest = hashlib.sha256()
    for array in [X0, X1, y, Xt0, Xt1, truth_test, samples, tree_seeds, inverse]:
        digest.update(np.ascontiguousarray(array).tobytes())
    return dict(
        condition=condition,
        replicate=replicate,
        input_hash=digest.hexdigest(),
        outcomes=rows,
        contrasts=contrasts,
        checks=dict(checks),
    )


def interval(values, family=1):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Inference requires at least two complete replicates")
    mean, se = values.mean(), values.std(ddof=1) / np.sqrt(len(values))
    radius = t.ppf(1 - 0.05 / (2 * family), len(values) - 1) * se
    return dict(mean=mean, mcse=se, lower=mean - radius, upper=mean + radius)


def summarize(output, config):
    units = [
        json.loads((output / "units" / f"{c:02}_{r:04}.json").read_text())
        for c in range(len(config["conditions"]))
        for r in range(config["repetitions"])
    ]
    outcomes = pd.DataFrame(row for unit in units for row in unit["outcomes"])
    contrasts = pd.DataFrame(row for unit in units for row in unit["contrasts"])
    keys = ["context", "n", "eta", "depth"]
    expected = len(units) * 2 * len(config["depths"]) * len(ARMS)
    if (
        len(outcomes) != expected
        or outcomes.duplicated(keys + ["replicate", "representation", "arm"]).any()
    ):
        raise AssertionError("Missing or duplicate experimental cells")
    family = len(config["conditions"]) * len(config["depths"]) * len(PRIMARY)
    estimates = []
    for names, frame in contrasts.groupby(keys + ["contrast"], sort=True):
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
    outcomes.groupby(keys + ["representation", "arm"]).agg(
        accuracy=("accuracy", "mean"),
        mcse=("accuracy", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        observed_accuracy=("observed_accuracy", "mean"),
        leaves=("leaves", "mean"),
        signal_root=("signal_root", "mean"),
        signal_peak=("signal_peak", "mean"),
        signal_union=("signal_union", "mean"),
        natural_fragmented_groups=("natural_fragmented_groups", "mean"),
        column_refit_disagreement=("column_refit_disagreement", "mean"),
        column_refit_accuracy_change=("column_refit_accuracy_change", "mean"),
    ).reset_index().to_csv(output / "summary.csv", index=False)
    checks = sum((Counter(unit["checks"]) for unit in units), Counter())
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

    labels = {
        "gaussian": "Gaussian noise",
        "binary": "Binary noise",
        "mixed": "Mixed noise",
        "correlated": "Correlated noise",
        "xor": "XOR + Gaussian noise",
        "majority": "Majority + Gaussian noise",
        "breast_cancer": "Breast-cancer covariates",
        "wine": "Wine covariates",
        "digits": "Digits covariates",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax, contrast, title in zip(
        axes,
        ["fragmentation_jittered", "interaction"],
        [
            "Removing artificial vote fragmentation",
            "Additional grouping benefit after jitter",
        ],
    ):
        for depth, offset, color in [(2, -0.12, "#23678a"), (5, 0.12, "#a04b30")]:
            rows = frame[
                (frame.n == 200)
                & (frame.eta == 0.4)
                & (frame.depth == depth)
                & (frame.contrast == contrast)
            ]
            rows = rows.set_index("context").reindex(
                [c for c in CONTEXTS if c in rows.context.values]
            )
            y = np.arange(len(rows)) + offset
            ax.errorbar(
                rows["mean"],
                y,
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
    present = [c for c in CONTEXTS if c in frame.context.values]
    axes[0].set_yticks(np.arange(len(present)), [labels[c] for c in present])
    axes[0].invert_yaxis()
    axes[1].legend(frameon=False)
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(output / f"effects.{suffix}", dpi=180, bbox_inches="tight")
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
    from cart_study.ablation import compute_unit

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/ablation"))
    parser.add_argument("--seed", type=int, default=20261008)
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
    conditions = [(c, n, eta) for c in CONTEXTS for n, eta in REGIMES]
    conditions += [(c, 200, 0.5) for c in ["gaussian", "mixed", "wine"]]
    config = dict(
        seed=args.seed,
        repetitions=2 if args.smoke else args.repetitions,
        test_size=200 if args.smoke else 5000,
        pool_size=20,
        depths=[2, 5],
        min_leaf=10,
        conditions=conditions,
        unit="Independent training/test draw and proposal randomness; all arms paired",
        estimand=(
            "Mean paired accuracy change " "separately within each context/regime/depth"
        ),
        primary=PRIMARY,
        multiplicity="Bonferroni over all primary cell contrasts; paired t intervals",
        risk=(
            "Integrate test-label flips; independently "
            "draw test covariates per replicate"
        ),
        representation="All constructed bits: Z versus Z + .25 U, U uniform[-1,1]",
        identity=(
            "Natural, canonical, or one threshold " "per tree inside known support gaps"
        ),
        reconstruction=(
            "Common partition candidates and priorities; global "
            "votes, no cutoff or impurity weighting"
        ),
        oracle=(
            "Known gaps for all constructed bits, "
            "including noise; other thresholds kept distinct"
        ),
        controls=(
            "Exact fixed-pool equivariance; CART "
            "refit column sensitivity measured separately"
        ),
        scope=(
            "Real covariates are empirical nuisance "
            "distributions; their outcomes are never used"
        ),
        missing=(
            "Any failed fit/check aborts; no " "dropped units; exact-config resume only"
        ),
        prior_information=(
            "Weak binary gain and population XOR "
            "results informed this design; not preregistered"
        ),
        versions=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            scipy=scipy.__version__,
            sklearn=sklearn.__version__,
            pandas=pd.__version__,
        ),
        source_hashes={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), Path(__file__).with_name("trees.py")]
        },
    )
    serialized = json.dumps(config, indent=2) + "\n"
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "config.json"
    if config_path.exists() and config_path.read_text() != serialized:
        raise ValueError(
            "Existing output has a different design or source; choose another output"
        )
    config_path.write_text(serialized)
    (args.output / "units").mkdir(exist_ok=True)

    Parallel(n_jobs=args.jobs, verbose=10)(
        delayed(compute_unit)(args.output, config, c, r)
        for c in range(len(config["conditions"]))
        for r in range(config["repetitions"])
    )
    summarize(args.output, config)


if __name__ == "__main__":
    main()
