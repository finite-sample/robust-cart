"""Compare tree-growing rules across specified populations and training samples."""

import argparse
import copy
import hashlib
import json
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from cart_study.baselines import ALOOFTree, ErrorTree, NodeVoteTree
from cart_study.design import (
    ALPHAS,
    REAL,
    SCALES,
    TAUS,
    dataset,
    generate,
    scaled_eta,
    scenario_id,
    scenarios,
)
from cart_study.trees import (
    HoldoutAblationTree,
    Node,
    VectorizedConsensusTree,
    cv_folds,
    development_split,
    fit_pool,
    fixed_split_predict,
    leaf_count,
    root_feature,
    split_loss,
    split_set,
    split_support,
)

ROOT = Path(__file__).resolve().parents[1]
SIZE_GRID = [(2, 20), (2, 5), (5, 20), (5, 5)]
STRONG_GRID = [(d, m, a) for d in (2, 5, 10, None) for m in (20, 5, 1) for a in ALPHAS]
DEFAULTS = dict(seeds=20, seed_start=2000, test_size=10000, proposals=100, pool_size=20)


def load_unit(spec, seed, permutation, settings):
    family = spec["family"]
    if family == "real":
        X, y = dataset(spec["name"], seed)
        train, test = development_split(y, seed)
        Xt, yt, Xv, yv = X[train], y[train], X[test], y[test]
    elif family == "continuous_xor":
        X, y = dataset("xor", seed)
        train, test = development_split(y, seed)
        Xt, yt, Xv, yv = X[train], y[train], X[test], y[test]
    else:
        Xt, yt = generate(spec, spec["n"], seed)
        Xv, yv = generate(spec, settings["test_size"], seed + 1000000)
        train = np.arange(len(yt))
        test = np.arange(len(yv)) + len(yt)
    columns = np.arange(Xt.shape[1])
    if permutation >= 0:
        columns = np.random.default_rng(seed * 31 + permutation + 50000).permutation(
            columns
        )
    metadata = dict(
        training_ids=train.tolist(),
        test_ids=test.tolist(),
        columns=columns.tolist(),
        training_hash=hashlib.sha256(Xt.tobytes() + yt.tobytes()).hexdigest(),
        test_hash=hashlib.sha256(Xv.tobytes() + yv.tobytes()).hexdigest(),
    )
    return Xt[:, columns], yt, Xv[:, columns], yv, metadata


def cap_tree(model, depth):
    result = copy.copy(model)

    def truncate(node, level):
        if level == depth or node.feature < 0:
            return Node(node.prediction, node.count)
        return Node(
            node.prediction,
            node.count,
            node.feature,
            node.threshold,
            truncate(node.left, level + 1),
            truncate(node.right, level + 1),
        )

    result.tree_ = truncate(model.tree_, 0)
    result.max_depth = depth
    return result


def pool_splits(pool, depth):
    result = []
    for model in pool.trees:
        tree, splits, stack = model.tree_, set(), [(0, 0)]
        while stack:
            node, level = stack.pop()
            if level >= depth or tree.feature[node] < 0:
                continue
            splits.add((int(tree.feature[node]), float(tree.threshold[node])))
            stack.extend(
                [
                    (tree.children_left[node], level + 1),
                    (tree.children_right[node], level + 1),
                ]
            )
        result.append(splits)
    return result


def candidates(X, y, seed, settings, binary_lookahead=False, only=None):
    """Yield a stable candidate grid; reuse fits across caps and support cutoffs."""

    def wanted(name):
        return only is None or name in only

    cart_cache = {}
    for name, grid in (
        ("cart_tuned", [(d, m, a) for d, m in SIZE_GRID for a in ALPHAS]),
        ("cart_strong", STRONG_GRID),
    ):
        if not wanted(name):
            continue
        for depth, leaf, alpha in grid:
            key = depth, leaf, alpha
            if key not in cart_cache:
                cart_cache[key] = DecisionTreeClassifier(
                    max_depth=depth,
                    min_samples_leaf=leaf,
                    ccp_alpha=alpha,
                    random_state=seed,
                ).fit(X, y)
            yield name, dict(depth=depth, leaf=leaf, alpha=alpha), cart_cache[key]
    constructors = {
        "holdout_selected": (HoldoutAblationTree, dict(scoring="selected")),
        "holdout_common": (HoldoutAblationTree, dict(scoring="common")),
        "holdout_refined": (
            HoldoutAblationTree,
            dict(scoring="selected", random_ties=True, keep_single_class=True),
        ),
        "node_vote": (NodeVoteTree, {}),
        "error_greedy": (ErrorTree, {}),
    }
    if len(np.unique(y)) <= 2:
        constructors["aloof"] = (ALOOFTree, {})
    if binary_lookahead:
        constructors["lookahead"] = (ErrorTree, dict(lookahead=True))
    for name, (cls, options) in constructors.items():
        if not wanted(name):
            continue
        options = dict(options)
        if issubclass(cls, (HoldoutAblationTree, NodeVoteTree)):
            options["n_subsamples"] = settings["proposals"]
        fitted = {}
        for depth, leaf in SIZE_GRID:
            if name == "lookahead":
                model = cls(
                    max_depth=depth,
                    min_samples_leaf=leaf,
                    random_state=seed,
                    **options,
                ).fit(X, y)
                yield name, dict(depth=depth, leaf=leaf), model
                continue
            if leaf not in fitted:
                fitted[leaf] = cls(
                    max_depth=5, min_samples_leaf=leaf, random_state=seed, **options
                ).fit(X, y)
            model = fitted[leaf] if depth == 5 else cap_tree(fitted[leaf], depth)
            yield name, dict(depth=depth, leaf=leaf), model
    for kind in ("bootstrap", "subsample"):
        pool_names = [
            f"{prefix}_{kind}"
            for prefix in ("frequency", "relative", "weighted", "relative_random")
        ]
        if not any(wanted(name) for name in pool_names):
            continue
        pools = {}
        for depth, leaf in SIZE_GRID:
            if leaf not in pools:
                pools[leaf] = fit_pool(
                    X,
                    y,
                    kind,
                    settings["pool_size"],
                    seed,
                    max_depth=5,
                    min_samples_leaf=leaf,
                )
            support = split_support(pool_splits(pools[leaf], depth))
            maximum = max(support.values(), default=1)
            for prefix in ("frequency", "relative", "weighted", "relative_random"):
                name = f"{prefix}_{kind}"
                if not wanted(name):
                    continue
                relative = prefix.startswith("relative")
                supplied = (
                    {s: f / maximum for s, f in support.items()}
                    if relative
                    else support
                )
                for tau in TAUS:
                    for scale in (SCALES if prefix == "weighted" else (0.0,)):
                        eta = scaled_eta(scale, support, tau, settings["pool_size"])
                        model = VectorizedConsensusTree(
                            supplied,
                            tau=tau,
                            eta=eta,
                            random_ties=prefix == "relative_random",
                            max_depth=depth,
                            min_samples_leaf=leaf,
                            random_state=seed,
                        ).fit(X, y)
                        yield name, dict(
                            depth=depth, leaf=leaf, tau=tau, scale=scale
                        ), model


def fixed_proposals(X, y, Xtest, ytest, seed):
    """Compare three scores on the same two binary-feature candidates."""
    rng = np.random.default_rng(seed + 7000000)
    fitting, evaluating = np.array_split(rng.permutation(len(y)), 2)
    splits = [(j, 0.5) for j in range(2)]
    scores = {key: [] for key in ("gini", "training_error", "holdout_error")}
    predictions = []
    for split in splits:
        pred = fixed_split_predict(X[fitting], y[fitting], Xtest, *split)
        predictions.append(pred)
        scores["gini"].append(split_loss(X[fitting], y[fitting], *split))
        train_pred = fixed_split_predict(X[fitting], y[fitting], X[fitting], *split)
        valid_pred = fixed_split_predict(X[fitting], y[fitting], X[evaluating], *split)
        scores["training_error"].append(np.mean(train_pred != y[fitting]))
        scores["holdout_error"].append(np.mean(valid_pred != y[evaluating]))
    result = {}
    for score, values in scores.items():
        winner = int(np.argmin(values))
        result[f"stump_{score}"] = (
            predictions[winner],
            dict(feature=winner, scores=list(map(float, values))),
        )
    return result


def evaluate(spec, seed, permutation, settings, only=None):
    threadpool_limits(1)
    X, y, Xtest, ytest, metadata = load_unit(spec, seed, permutation, settings)
    lookahead = spec["family"] == "interaction"
    folds = cv_folds(y, seed)
    metadata["folds"] = [dict(train=a.tolist(), valid=b.tolist()) for a, b in folds]
    scores, parameters, counts = {}, {}, {}
    started = time.perf_counter()
    for fold, (train, valid) in enumerate(folds):
        positions = {}
        for name, params, model in candidates(
            X[train], y[train], seed + fold, settings, lookahead, only
        ):
            index = positions.get(name, 0)
            positions[name] = index + 1
            if fold == 0:
                scores.setdefault(name, []).append(0)
                parameters.setdefault(name, []).append(params)
            assert parameters[name][index] == params
            scores[name][index] += int(np.sum(model.predict(X[valid]) != y[valid]))
        if fold == 0:
            counts = positions
        assert positions == counts
    tune_seconds = time.perf_counter() - started
    selected = {name: int(np.argmin(loss)) for name, loss in scores.items()}
    depth_selected = {}
    for name, configs in parameters.items():
        for depth in sorted({p["depth"] for p in configs if p["depth"] is not None}):
            ids = [i for i, p in enumerate(configs) if p["depth"] == depth]
            depth_selected[name, depth] = min(ids, key=lambda i: (scores[name][i], i))
    predictions, rows, curves, positions = {}, [], [], {}

    def record(name, model, pred, params, validation_error=None):
        predictions[name] = np.asarray(pred)
        rows.append(
            dict(
                method=name,
                accuracy=float(np.mean(pred == ytest)),
                leaves=leaf_count(model) if model is not None else 2,
                parameters=params,
                validation_error=validation_error,
                root=(
                    root_feature(model) if model is not None else params.get("feature")
                ),
                splits=sorted(split_set(model)) if model is not None else [],
            )
        )

    for name, params, model in candidates(X, y, seed + 50, settings, lookahead, only):
        index = positions.get(name, 0)
        positions[name] = index + 1
        is_selected = index == selected[name]
        is_curve = index == depth_selected.get((name, params["depth"]))
        if is_selected or is_curve:
            pred = model.predict(Xtest)
            if is_selected:
                record(name, model, pred, params, scores[name][index] / len(y))
            if is_curve:
                curves.append(
                    dict(
                        method=name,
                        depth=params["depth"],
                        leaves=leaf_count(model),
                        accuracy=float(np.mean(pred == ytest)),
                        parameters=params,
                    )
                )
    if only is None:
        base = DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=5, random_state=seed
        ).fit(X, y)
        record("cart", base, base.predict(Xtest), dict(depth=5, leaf=5))
        for name, random_ties, keep in (
            ("fixed_original", False, False),
            ("fixed_random_ties", True, False),
            ("fixed_keep_single_class", False, True),
            ("fixed_refined", True, True),
        ):
            model = HoldoutAblationTree(
                random_ties=random_ties,
                keep_single_class=keep,
                max_depth=5,
                min_samples_leaf=5,
                random_state=seed,
                n_subsamples=settings["proposals"],
            ).fit(X, y)
            record(name, model, model.predict(Xtest), dict(depth=5, leaf=5))
        if spec["family"] == "objective":
            for name, (pred, params) in fixed_proposals(
                X, y, Xtest, ytest, seed
            ).items():
                record(name, None, pred, params)
    unit = dict(
        spec=spec,
        seed=seed,
        permutation=permutation,
        data=metadata,
        rows=rows,
        curves=curves,
        cv_errors=scores,
        candidate_parameters=parameters,
        tuning_seconds=tune_seconds,
        total_seconds=time.perf_counter() - started,
    )
    return unit, dict(y=ytest, **predictions)


def all_tasks(settings):
    tasks = []
    for spec in scenarios():
        for seed in range(
            settings["seed_start"], settings["seed_start"] + settings["seeds"]
        ):
            for permutation in range(3 if spec["family"] == "interaction" else 1):
                tasks.append((spec, seed, permutation))
    for name in REAL:
        for seed in range(1000, 1005):
            tasks.append((dict(family="real", name=name), seed, 0))
    for seed in range(1000, 1005):
        for permutation in (-1, 0, 1, 2):
            tasks.append(
                (dict(family="continuous_xor", n=600, p=20), seed, permutation)
            )
    return tasks


def unit_id(spec, seed, permutation):
    return f"{scenario_id(spec)}_seed-{seed}_perm-{permutation}"


def summarize(output):
    rows, curves = [], []
    for path in sorted((output / "units").glob("*.json")):
        unit = json.loads(path.read_text())
        prefix = dict(
            unit=path.stem,
            scenario=scenario_id(unit["spec"]),
            **unit["spec"],
            seed=unit["seed"],
            permutation=unit["permutation"],
        )
        for row in unit["rows"]:
            rows.append(
                {
                    **prefix,
                    **{
                        k: v
                        for k, v in row.items()
                        if k not in ("splits", "parameters")
                    },
                    "parameters": json.dumps(row["parameters"], sort_keys=True),
                }
            )
        for row in unit["curves"]:
            curves.append(
                {**prefix, **{k: v for k, v in row.items() if k != "parameters"}}
            )
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(curves).to_csv(output / "curves.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/study")
    parser.add_argument("--jobs", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument(
        "--tasks", type=Path, help="Explicit confirmation tasks as JSON"
    )
    args = parser.parse_args()
    if args.summarize:
        summarize(args.output)
        return
    settings = dict(DEFAULTS)
    tasks = all_tasks(settings)
    if args.pilot:
        tasks = [
            (dict(family="objective", n=800, p=2, mixture=0.0), 42, 0),
            (dict(family="noise", n=800, p=20, noise=0.4), 42, 0),
            (dict(family="interaction", n=800, p=20, mixture=0.0), 42, 0),
            (dict(family="real", name="digits"), 42, 0),
        ]
    if args.smoke:
        settings.update(proposals=3, pool_size=3, test_size=100)
        tasks = [(dict(family="objective", n=40, p=2, mixture=0.0), 42, 0)]
    only = None
    if args.tasks:
        confirmation = json.loads(args.tasks.read_text())
        tasks = [
            (t["spec"], t["seed"], t["permutation"]) for t in confirmation["tasks"]
        ]
        only = confirmation["methods"]
    config = dict(
        settings=settings,
        tasks=tasks,
        methods=only,
        size_grid=SIZE_GRID,
        strong_grid=STRONG_GRID,
        taus=TAUS,
        scales=SCALES,
        python=platform.python_version(),
        numpy=np.__version__,
        sklearn=sklearn.__version__,
        source_hashes={
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "cart_study/study.py",
                "cart_study/baselines.py",
                "cart_study/trees.py",
                "cart_study/design.py",
            )
        },
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "units").mkdir(exist_ok=True)
    serialized = json.dumps(config, indent=2) + "\n"
    manifest = args.output / "config.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        previous.pop("source_update", None)
        if previous != json.loads(serialized):
            raise ValueError("Configuration changed; choose another output directory")
    else:
        manifest.write_text(serialized)
    pending = [
        (spec, seed, perm)
        for spec, seed, perm in tasks
        if not (args.output / "units" / f"{unit_id(spec, seed, perm)}.json").exists()
    ]
    started = time.perf_counter()
    with ProcessPoolExecutor(args.jobs) as executor:
        futures = {
            executor.submit(evaluate, spec, seed, perm, settings, only): unit_id(
                spec, seed, perm
            )
            for spec, seed, perm in pending
        }
        for i, future in enumerate(as_completed(futures), 1):
            unit, predictions = future.result()
            name = futures[future]
            np.savez_compressed(args.output / "units" / f"{name}.npz", **predictions)
            (args.output / "units" / f"{name}.json").write_text(
                json.dumps(unit, indent=2) + "\n"
            )
            print(
                f"{i}/{len(pending)} {name} ({time.perf_counter() - started:.1f}s)",
                flush=True,
            )
    summarize(args.output)
    (args.output / "complete.json").write_text(
        json.dumps(dict(units=len(tasks), complete=True)) + "\n"
    )


if __name__ == "__main__":
    main()
