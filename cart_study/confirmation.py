"""Evaluate screened comparisons on independent samples and perturb settings."""

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from cart_study.design import scaled_eta
from cart_study.report import paired_interval, per_dataset
from cart_study.study import (
    ROOT,
    cap_tree,
    load_unit,
    pool_splits,
    scenario_id,
    unit_id,
)
from cart_study.trees import (
    HoldoutAblationTree,
    VectorizedConsensusTree,
    fit_pool,
    leaf_count,
    split_support,
)


def confirmation_checks(metrics, eligible):
    datasets = per_dataset(metrics)
    rows = []
    for entry in eligible:
        scenario, method = scenario_id(entry["spec"]), entry["method"]
        case = datasets[datasets.scenario == scenario]
        wide = case.pivot(index="seed", columns="method", values="accuracy")
        if set(wide.index) != set(range(3000, 3020)):
            raise ValueError(
                "Confirmation needs exactly 20 independent simulation replicates"
            )
        for comparator in ("cart_strong", entry["comparator"]):
            difference = wide[method] - wide[comparator]
            mean, low, high = paired_interval(difference)
            passed = low > 0 and (difference > 1e-12).sum() >= 16
            if entry["spec"]["family"] == "interaction":
                for _, group in metrics[metrics.scenario == scenario].groupby(
                    "permutation"
                ):
                    means = group.groupby("method").accuracy.mean()
                    passed = passed and means[method] > means[comparator]
            rows.append(
                dict(
                    scenario=scenario,
                    method=method,
                    comparator=comparator,
                    difference=mean,
                    low=low,
                    high=high,
                    wins=int((difference > 1e-12).sum()),
                    passed=bool(passed),
                )
            )
    return pd.DataFrame(rows)


def perturbations(params, settings):
    for name, key, value in (
        ("shallower", "depth", max(1, params["depth"] - 1)),
        ("deeper", "depth", params["depth"] + 1),
        ("smaller_leaves", "leaf", max(1, round(params["leaf"] * 0.8))),
        ("larger_leaves", "leaf", max(1, round(params["leaf"] * 1.2))),
    ):
        yield name, {**params, key: value}, dict(settings)
    for name, factor in (("fewer_resamples", 0.5), ("more_resamples", 2)):
        changed = dict(settings)
        for key in ("proposals", "pool_size"):
            changed[key] = max(1, round(settings[key] * factor))
        yield name, dict(params), changed


def fit_method(method, X, y, seed, params, settings):
    depth, leaf = params["depth"], params["leaf"]
    if method.startswith("holdout_"):
        refined = method == "holdout_refined"
        model = HoldoutAblationTree(
            scoring="common" if method == "holdout_common" else "selected",
            random_ties=refined,
            keep_single_class=refined,
            n_subsamples=settings["proposals"],
            max_depth=max(5, depth),
            min_samples_leaf=leaf,
            random_state=seed,
        ).fit(X, y)
        return cap_tree(model, depth) if depth < 5 else model
    prefix, kind = method.rsplit("_", 1)
    if prefix not in {"frequency", "relative", "relative_random", "weighted"}:
        raise ValueError(f"Unknown screened method: {method}")
    pool = fit_pool(
        X,
        y,
        kind,
        settings["pool_size"],
        seed,
        max_depth=max(5, depth),
        min_samples_leaf=leaf,
    )
    support = split_support(pool_splits(pool, depth))
    maximum = max(support.values(), default=1)
    supplied = (
        {s: f / maximum for s, f in support.items()}
        if prefix.startswith("relative")
        else support
    )
    eta = scaled_eta(params["scale"], support, params["tau"], settings["pool_size"])
    return VectorizedConsensusTree(
        supplied,
        tau=params["tau"],
        eta=eta,
        random_ties=prefix == "relative_random",
        max_depth=depth,
        min_samples_leaf=leaf,
        random_state=seed,
    ).fit(X, y)


def perturb_unit(path, methods, settings):
    threadpool_limits(1)
    unit = json.loads(path.read_text())
    X, y, Xt, yt, _ = load_unit(
        unit["spec"], unit["seed"], unit["permutation"], settings
    )
    baseline = next(r["accuracy"] for r in unit["rows"] if r["method"] == "cart_strong")
    rows, predictions = [], {"y": yt}
    started = time.perf_counter()
    for chosen in unit["rows"]:
        method = chosen["method"]
        if method not in methods:
            continue
        for name, params, perturbed in perturbations(chosen["parameters"], settings):
            model = fit_method(method, X, y, unit["seed"] + 50, params, perturbed)
            prediction = model.predict(Xt)
            predictions[f"{method}__{name}"] = prediction
            accuracy = float(np.mean(prediction == yt))
            rows.append(
                dict(
                    scenario=scenario_id(unit["spec"]),
                    seed=unit["seed"],
                    permutation=unit["permutation"],
                    method=method,
                    perturbation=name,
                    parameters=params,
                    settings=perturbed,
                    accuracy=accuracy,
                    leaves=leaf_count(model),
                    difference=accuracy - baseline,
                )
            )
    return dict(rows=rows, seconds=time.perf_counter() - started), predictions


def perturbation_summary(rows):
    values = rows.groupby(
        ["scenario", "method", "perturbation", "seed"], as_index=False
    ).difference.mean()
    summary = []
    for keys, group in values.groupby(["scenario", "method", "perturbation"]):
        mean, low, high = paired_interval(group.difference)
        summary.append(
            dict(
                zip(("scenario", "method", "perturbation"), keys),
                difference=mean,
                low=low,
                high=high,
                datasets=len(group),
                passed=mean > 1e-12,
            )
        )
    return pd.DataFrame(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("results/study"))
    parser.add_argument(
        "--confirmation", type=Path, default=Path("results/confirmation")
    )
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    design = json.loads((args.study / "confirmation_tasks.json").read_text())
    if not design["eligible"]:
        print("No method-condition pair passed the screen; no confirmation candidates.")
        return
    complete = json.loads((args.confirmation / "complete.json").read_text())
    if not complete["complete"] or complete["units"] != len(design["tasks"]):
        raise ValueError("Confirmation fits are incomplete")
    config = json.loads((args.confirmation / "config.json").read_text())
    for name, expected in config["source_hashes"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Confirmation source changed: {name}")
    checks = confirmation_checks(
        pd.read_csv(args.confirmation / "metrics.csv"), design["eligible"]
    )
    checks.to_csv(args.confirmation / "checks.csv", index=False)
    successful = checks[(checks.comparator == "cart_strong") & checks.passed]
    output = args.confirmation / "perturbations"
    output.mkdir(exist_ok=True)
    settings = config["settings"]
    manifest = dict(
        settings=settings,
        methods=successful[["scenario", "method"]].to_dict("records"),
        source_hashes={
            **config["source_hashes"],
            "cart_study/confirmation.py": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
    )
    serialized = json.dumps(manifest, indent=2) + "\n"
    config_path = output / "config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        previous.pop("source_update", None)
        if previous != json.loads(serialized):
            raise ValueError("Perturbation configuration changed")
    else:
        config_path.write_text(serialized)
    tasks = []
    for task in design["tasks"]:
        scenario = scenario_id(task["spec"])
        methods = successful[successful.scenario == scenario].method.tolist()
        name = unit_id(task["spec"], task["seed"], task["permutation"])
        if methods and not (output / f"{name}.json").exists():
            tasks.append((args.confirmation / "units" / f"{name}.json", methods))
    with ProcessPoolExecutor(args.jobs) as executor:
        futures = {
            executor.submit(perturb_unit, path, methods, settings): path.stem
            for path, methods in tasks
        }
        for i, future in enumerate(as_completed(futures), 1):
            unit, predictions = future.result()
            name = futures[future]
            np.savez_compressed(output / f"{name}.npz", **predictions)
            (output / f"{name}.json").write_text(json.dumps(unit, indent=2) + "\n")
            print(f"Perturbations {i}/{len(tasks)}", flush=True)
    rows = [
        r
        for p in output.glob("family-*.json")
        for r in json.loads(p.read_text())["rows"]
    ]
    passed_pairs = []
    if rows:
        pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
        summary = perturbation_summary(pd.DataFrame(rows))
        summary.to_csv(output / "summary.csv", index=False)
        for (scenario, method), group in summary.groupby(["scenario", "method"]):
            if len(group) == 6 and (group.datasets == 20).all() and group.passed.all():
                passed_pairs.append(dict(scenario=scenario, method=method))
    (args.confirmation / "passed_checks.json").write_text(
        json.dumps(passed_pairs, indent=2) + "\n"
    )
    print(
        f"{len(passed_pairs)} method-condition pairs pass "
        "confirmation and perturbations."
    )


if __name__ == "__main__":
    main()
