"""Measure accuracy and stability across repeated training subsets."""

import argparse
import hashlib
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from cart_study.design import (
    ALPHAS,
    CONSENSUS_SCENARIOS,
    NODE_SYNTHETIC,
    REAL,
    SCALES,
    TAUS,
    dataset,
    scaled_eta,
)
from cart_study.trees import (
    ConsensusTree,
    HoldoutTree,
    cv_folds,
    development_split,
    fit_pool,
    leaf_count,
    mean_jaccard,
    prediction_disagreement,
    root_agreement,
    root_feature,
    split_set,
    split_support,
)


@dataclass(frozen=True)
class Settings:
    """Fixed experiment settings, written into every result file."""

    seeds: int = 5
    refits: int = 10
    subsamples: int = 100
    pool_size: int = 20
    max_depth: int = 5
    min_samples_leaf: int = 5
    seed_start: int = 1000

    @property
    def tree_options(self):
        return {
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
        }


def tune_pruning(X, y, seed, settings):
    losses = np.zeros(len(ALPHAS))
    for tr, va in cv_folds(y, seed):
        for i, alpha in enumerate(ALPHAS):
            tree = DecisionTreeClassifier(
                random_state=seed, ccp_alpha=alpha, **settings.tree_options
            ).fit(X[tr], y[tr])
            losses[i] += np.sum(tree.predict(X[va]) != y[va])
    return ALPHAS[int(np.argmin(losses))]


def tune_pool_methods(X, y, kind, seed, settings):
    """Fit each fold's pool once, exclusively on that fold's training rows."""
    frequency_loss = np.zeros(len(TAUS))
    relative_loss = np.zeros(len(TAUS))
    weighted_loss = np.zeros((len(TAUS), len(SCALES)))
    for fold, (train, valid) in enumerate(cv_folds(y, seed)):
        pool = fit_pool(
            X[train],
            y[train],
            kind,
            settings.pool_size,
            seed + fold,
            **settings.tree_options,
        )
        support = split_support(pool.split_sets)
        for i, tau in enumerate(TAUS):
            tree = ConsensusTree(support, tau=tau, **settings.tree_options)
            tree.fit(X[train], y[train])
            frequency_loss[i] += np.sum(tree.predict(X[valid]) != y[valid])
            relative_support = {
                s: f / max(support.values()) for s, f in support.items()
            }
            tree = ConsensusTree(relative_support, tau=tau, **settings.tree_options)
            tree.fit(X[train], y[train])
            relative_loss[i] += np.sum(tree.predict(X[valid]) != y[valid])
            for j, scale in enumerate(SCALES):
                eta = scaled_eta(scale, support, tau, settings.pool_size)
                tree = ConsensusTree(support, tau=tau, eta=eta, **settings.tree_options)
                tree.fit(X[train], y[train])
                weighted_loss[i, j] += np.sum(tree.predict(X[valid]) != y[valid])
    i, j = np.unravel_index(np.argmin(weighted_loss), weighted_loss.shape)
    return {
        "frequency_tau": TAUS[int(np.argmin(frequency_loss))],
        "weighted_tau": TAUS[i],
        "weighted_scale": SCALES[j],
        "relative_tau": TAUS[int(np.argmin(relative_loss))],
    }


def evaluate_unit(name, seed, settings):
    """One dataset/split, with repeated model building on distinct training subsets."""
    X, y = dataset(name, seed)
    if not np.isfinite(X).all():
        raise ValueError("Unexpected missing or nonfinite feature")
    development, test = development_split(y, seed)
    rng = np.random.default_rng(seed + 100000)
    refit_rows, metric_rows, predictions, structures, roots = [], [], {}, {}, {}
    samples = []

    def record(
        method, model, prediction, refit, seconds, parameters, kind="single_tree"
    ):
        predictions.setdefault(method, []).append(prediction)
        row = {
            "scenario": name,
            "dataset_seed": seed,
            "refit": refit,
            "method": method,
            "object": kind,
            "accuracy": float(np.mean(prediction == y[test])),
            "seconds": seconds,
            "parameters": json.dumps(parameters, sort_keys=True),
            "leaves": None,
        }
        if model is not None:
            structures.setdefault(method, []).append(split_set(model))
            roots.setdefault(method, []).append(root_feature(model))
            row["leaves"] = leaf_count(model)
        refit_rows.append(row)

    for refit in range(settings.refits):
        # Distinct identities avoid leakage between inner folds via bootstrap copies.
        chosen, _ = train_test_split(
            development,
            train_size=0.8,
            stratify=y[development],
            random_state=int(rng.integers(2**31)),
        )
        samples.append(chosen.tolist())
        Xt, yt = X[chosen], y[chosen]
        fit_seed = seed * 100 + refit
        start = time.perf_counter()
        base = DecisionTreeClassifier(
            random_state=fit_seed, **settings.tree_options
        ).fit(Xt, yt)
        record(
            "cart", base, base.predict(X[test]), refit, time.perf_counter() - start, {}
        )
        start = time.perf_counter()
        alpha = tune_pruning(Xt, yt, fit_seed, settings)
        pruned = DecisionTreeClassifier(
            random_state=fit_seed, ccp_alpha=alpha, **settings.tree_options
        ).fit(Xt, yt)
        record(
            "pruned_cart",
            pruned,
            pruned.predict(X[test]),
            refit,
            time.perf_counter() - start,
            {"alpha": alpha},
        )
        if name in NODE_SYNTHETIC + REAL:
            for scoring in ("selected", "common"):
                start = time.perf_counter()
                model = HoldoutTree(
                    scoring=scoring,
                    n_subsamples=settings.subsamples,
                    random_state=fit_seed,
                    **settings.tree_options,
                ).fit(Xt, yt)
                record(
                    "holdout_" + scoring,
                    model,
                    model.predict(X[test]),
                    refit,
                    time.perf_counter() - start,
                    {"node_diagnostics": model.diagnostics_},
                )
            continue
        for kind in ("bootstrap", "subsample"):
            start = time.perf_counter()
            selected = tune_pool_methods(Xt, yt, kind, fit_seed, settings)
            tune_seconds = time.perf_counter() - start
            start = time.perf_counter()
            pool = fit_pool(
                Xt, yt, kind, settings.pool_size, fit_seed, **settings.tree_options
            )
            pool_seconds = time.perf_counter() - start
            support = split_support(pool.split_sets)
            for method, tau, eta in (
                ("relative_frequency_tree", selected["relative_tau"], 0.0),
                ("frequency_tree", selected["frequency_tau"], 0.0),
                (
                    "weighted_tree",
                    selected["weighted_tau"],
                    scaled_eta(
                        selected["weighted_scale"],
                        support,
                        selected["weighted_tau"],
                        settings.pool_size,
                    ),
                ),
            ):
                start = time.perf_counter()
                candidate_support = support
                if method == "relative_frequency_tree":
                    candidate_support = {
                        s: f / max(support.values()) for s, f in support.items()
                    }
                model = ConsensusTree(
                    candidate_support, tau=tau, eta=eta, **settings.tree_options
                ).fit(Xt, yt)
                prediction = model.predict(X[test])
                seconds = tune_seconds + pool_seconds + time.perf_counter() - start
                record(
                    method + "_" + kind,
                    model,
                    prediction,
                    refit,
                    seconds,
                    dict(
                        selected, pool_seconds=pool_seconds, tuning_seconds=tune_seconds
                    ),
                )
    for method, preds in predictions.items():
        rows = [r for r in refit_rows if r["method"] == method]
        metrics = {
            "scenario": name,
            "dataset_seed": seed,
            "method": method,
            "object": rows[0]["object"],
            "accuracy": float(np.mean([r["accuracy"] for r in rows])),
            "prediction_disagreement": prediction_disagreement(preds),
            "seconds": float(np.mean([r["seconds"] for r in rows])),
            "leaves": None,
            "split_jaccard": None,
            "empty_pairs": None,
            "root_agreement": None,
            "unsplit_fraction": None,
        }
        if method in structures:
            jac, empty = mean_jaccard(structures[method])
            metrics.update(
                leaves=float(np.mean([r["leaves"] for r in rows])),
                split_jaccard=None if np.isnan(jac) else jac,
                empty_pairs=empty,
                root_agreement=root_agreement(roots[method]),
                unsplit_fraction=float(np.mean([r < 0 for r in roots[method]])),
            )
        metric_rows.append(metrics)
    return {
        "scenario": name,
        "seed": seed,
        "settings": asdict(settings),
        "data": {
            "n": len(y),
            "p": X.shape[1],
            "classes": np.unique(y, return_counts=True)[1].tolist(),
            "sha256": hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest(),
            "development_ids": development.tolist(),
            "test_ids": test.tolist(),
            "refit_ids": samples,
        },
        "refits": refit_rows,
        "metrics": metric_rows,
        "predictions": {k: np.asarray(v).tolist() for k, v in predictions.items()},
        "test_labels": y[test].tolist(),
        "structures": {k: [sorted(s) for s in sets] for k, sets in structures.items()},
        "roots": roots,
    }


def summarize(directory):
    directory = Path(directory)
    units = [
        json.loads(p.read_text()) for p in sorted((directory / "units").glob("*.json"))
    ]
    for key in ("refits", "metrics"):
        pd.DataFrame([r for unit in units for r in unit[key]]).to_csv(
            directory / f"{key}.csv", index=False
        )
    metrics = pd.DataFrame([r for unit in units for r in unit["metrics"]])
    for scope, scenarios in (
        ("node_synthetic", NODE_SYNTHETIC),
        ("node_real", REAL),
        ("consensus_synthetic", CONSENSUS_SCENARIOS),
    ):
        part = metrics[metrics.scenario.isin(scenarios)]
        if part.empty:
            continue
        summary = part.groupby("method")[
            ["accuracy", "prediction_disagreement", "leaves", "seconds"]
        ].mean()
        summary.to_csv(directory / f"summary_{scope}.csv")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/stability"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.output)
        return
    settings = (
        Settings()
        if not args.smoke
        else Settings(seeds=1, refits=2, subsamples=5, pool_size=3)
    )
    scenarios = NODE_SYNTHETIC + REAL + CONSENSUS_SCENARIOS
    if args.smoke:
        scenarios = ("low_noise", "wine", "small_sample")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "units").mkdir(exist_ok=True)
    config = {
        "settings": asdict(settings),
        "scenarios": scenarios,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "source_hashes": {
            name: hashlib.sha256(
                (Path(__file__).resolve().parents[1] / name).read_bytes()
            ).hexdigest()
            for name in (
                "cart_study/trees.py",
                "cart_study/stability.py",
                "cart_study/design.py",
            )
        },
        "grids": dict(tau=TAUS, scale=SCALES, alpha=ALPHAS),
        "outer_resampling": (
            "80% stratified without replacement; fixed original 30% test split"
        ),
        "jobs": args.jobs,
    }
    config_path = args.output / "config.json"
    serialized = json.dumps(config, indent=2) + "\n"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        previous.pop("source_update", None)
        if previous != json.loads(serialized):
            raise ValueError("Configuration changed; choose another output directory")
    else:
        config_path.write_text(serialized)
    tasks = []
    for name in scenarios:
        for seed in range(settings.seed_start, settings.seed_start + settings.seeds):
            path = args.output / "units" / f"{name}_{seed}.json"
            if not path.exists():
                tasks.append((name, seed, path))
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        pending = {
            executor.submit(evaluate_unit, name, seed, settings): path
            for name, seed, path in tasks
        }
        for i, future in enumerate(as_completed(pending), 1):
            path = pending[future]
            result = future.result()
            path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            print(
                f"{i}/{len(tasks)} {path.stem} ({time.perf_counter() - started:.1f}s)",
                flush=True,
            )
    summarize(args.output)
    (args.output / "complete.json").write_text(
        json.dumps(
            {"units": len(scenarios) * settings.seeds, "complete": True}, indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
