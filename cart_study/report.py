"""Derive figures, paired comparisons, and confirmation tasks from the study."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from cart_study.design import objective_population, scenario_id, scenarios
from cart_study.study import summarize

LABELS = {
    "cart": "CART, depth 5",
    "cart_tuned": "CART, common size grid",
    "cart_strong": "CART, wider size grid",
    "holdout_selected": "Selected holdout",
    "holdout_common": "Common-fold holdout",
    "holdout_refined": "Refined holdout",
    "aloof": "ALOOF",
    "node_vote": "Node-wise votes",
    "error_greedy": "Greedy accuracy",
    "lookahead": "Two-level lookahead",
    "relative_bootstrap": "Relative frequency, bootstrap",
    "relative_subsample": "Relative frequency, subsamples",
    "relative_random_bootstrap": "Relative bootstrap, random ties",
    "relative_random_subsample": "Relative subsamples, random ties",
    "weighted_bootstrap": "Impurity-weighted bootstrap",
    "weighted_subsample": "Impurity-weighted subsamples",
}
MAIN_METHODS = [
    "holdout_selected",
    "holdout_common",
    "holdout_refined",
    "relative_bootstrap",
    "relative_subsample",
    "relative_random_bootstrap",
    "weighted_bootstrap",
    "aloof",
    "node_vote",
    "lookahead",
]
PROPOSED = ["holdout_selected", "holdout_common", "holdout_refined"] + [
    f"{method}_{kind}"
    for method in ("frequency", "relative", "weighted", "relative_random")
    for kind in ("bootstrap", "subsample")
]


def paired_interval(difference):
    values = np.asarray(difference, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, np.nan, np.nan
    half = float(
        t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    )
    return mean, mean - half, mean + half


def per_dataset(metrics):
    return metrics.groupby(["scenario", "seed", "method"], as_index=False)[
        ["accuracy", "leaves"]
    ].mean()


def comparisons(metrics):
    values = per_dataset(metrics)
    rows = []
    for scenario, group in values.groupby("scenario"):
        wide = group.pivot(index="seed", columns="method", values="accuracy")
        for comparator in (
            "cart_strong",
            "cart_tuned",
            "aloof",
            "node_vote",
            "lookahead",
        ):
            if comparator not in wide:
                continue
            for method in wide:
                difference = (wide[method] - wide[comparator]).dropna()
                mean, low, high = paired_interval(difference)
                overlapping = scenario.startswith("family-real_")
                if overlapping:
                    low, high = np.nan, np.nan
                rows.append(
                    dict(
                        scenario=scenario,
                        method=method,
                        comparator=comparator,
                        datasets=len(difference),
                        replication_unit=(
                            "overlapping split" if overlapping else "dataset"
                        ),
                        difference=mean,
                        low=low,
                        high=high,
                        wins=int((difference > 1e-12).sum()),
                        ties=int((difference.abs() <= 1e-12).sum()),
                    )
                )
    return pd.DataFrame(rows)


def screen(metrics):
    """Apply the exploratory gate after averaging permutations within datasets."""
    values = per_dataset(metrics)
    checks, eligible = [], []
    specs = {scenario_id(s): s for s in scenarios()}
    for scenario, group in values.groupby("scenario"):
        if scenario not in specs:
            continue
        wide = group.pivot(index="seed", columns="method", values="accuracy")
        if len(wide) != 20:
            raise ValueError("The screen requires all 20 independent datasets")
        baselines = [m for m in ("aloof", "node_vote", "lookahead") if m in wide]
        best = max(baselines, key=lambda m: wide[m].mean())
        for method in PROPOSED:
            if method not in wide:
                continue
            passes = []
            for comparator in ("cart_strong", best):
                diff = wide[method] - wide[comparator]
                mean, low, high = paired_interval(diff)
                passed = low > 0 and (diff > 1e-12).sum() >= 16
                if comparator == "cart_strong":
                    passes.append(bool(passed))
                checks.append(
                    dict(
                        scenario=scenario,
                        method=method,
                        comparator=comparator,
                        difference=mean,
                        low=low,
                        high=high,
                        wins=int((diff > 1e-12).sum()),
                        passed=bool(passed),
                    )
                )
            if specs[scenario]["family"] == "interaction":
                original = metrics[metrics.scenario == scenario]
                for _, permutation in original.groupby("permutation"):
                    means = permutation.groupby("method").accuracy.mean()
                    passes.append(means[method] > means["cart_strong"])
            if all(passes):
                eligible.append(
                    dict(spec=specs[scenario], method=method, comparator=best)
                )
    return pd.DataFrame(checks), eligible


def confirmation_design(eligible):
    conditions = {scenario_id(e["spec"]): e["spec"] for e in eligible}
    tasks = []
    for spec in conditions.values():
        for seed in range(3000, 3020):
            for permutation in range(3 if spec["family"] == "interaction" else 1):
                tasks.append(dict(spec=spec, seed=seed, permutation=permutation))
    methods = sorted(
        {"cart_strong"}
        | {e["method"] for e in eligible}
        | {e["comparator"] for e in eligible}
    )
    return dict(eligible=eligible, methods=methods, tasks=tasks)


def noise_roots(output):
    rows = []
    for path in (output / "units").glob("*.json"):
        unit = json.loads(path.read_text())
        if unit["spec"]["family"] != "noise":
            continue
        for row in unit["rows"]:
            root = row["root"]
            rows.append(
                dict(
                    scenario=scenario_id(unit["spec"]),
                    seed=unit["seed"],
                    method=row["method"],
                    signal_root=root >= 0 and unit["data"]["columns"][root] == 0,
                    unsplit=root < 0,
                )
            )
    values = (
        pd.DataFrame(rows)
        .groupby(["scenario", "seed", "method"])[["signal_root", "unsplit"]]
        .mean()
    )
    return values.groupby(["scenario", "method"]).mean().reset_index()


def plotting():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cart-mpl"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def style():
    plt = plotting()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11,
            "figure.dpi": 140,
            "savefig.bbox": "tight",
            "axes.labelcolor": "#333333",
        }
    )


def save_figure(fig, output, name):
    plt = plotting()
    fig.savefig(output / f"{name}.png")
    fig.savefig(output / f"{name}.pdf")
    plt.close(fig)


def objective_figure(output):
    plt = plotting()
    mixtures = np.linspace(0, 1, 101)
    error, gini = np.zeros((2, len(mixtures))), np.zeros((2, len(mixtures)))
    for i, mixture in enumerate(mixtures):
        atoms, mass = objective_population(mixture)
        for feature in (0, 1):
            for side in (0, 1):
                subset = atoms[:, feature] == side
                w = mass[subset].sum()
                probability = mass[subset & (atoms[:, 2] == 1)].sum() / w
                error[feature, i] += w * min(probability, 1 - probability)
                gini[feature, i] += 2 * w * probability * (1 - probability)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), layout="constrained")
    for axis, values, title in zip(
        axes, (gini, 100 * error), ("Weighted child Gini", "Population stump error (%)")
    ):
        for feature, color in ((0, "#32658a"), (1, "#b55b37")):
            axis.plot(
                mixtures,
                values[feature],
                label=f"Split on {'AB'[feature]}",
                color=color,
                lw=2,
            )
        axis.set(xlabel="Mixture toward agreement", ylabel=title)
        axis.grid(alpha=0.15)
    axes[0].legend(frameon=False)
    fig.suptitle(
        "Gini and classification error can favor different splits",
        x=0.04,
        ha="left",
        fontsize=13,
    )
    save_figure(fig, output, "objective")


def illustrative_specs():
    return (
        [dict(family="objective", n=n, p=2, mixture=0.0) for n in (200, 800)]
        + [dict(family="noise", n=n, p=20, noise=0.4) for n in (200, 800)]
        + [dict(family="interaction", n=n, p=20, mixture=0.0) for n in (200, 800)]
    )


def accuracy_figure(paired, output):
    plt = plotting()
    fig, axes = plt.subplots(
        3, 2, figsize=(11, 11), sharey=True, sharex="row", layout="constrained"
    )
    for axis, spec in zip(axes.flat, illustrative_specs()):
        case = paired[
            (paired.scenario == scenario_id(spec))
            & (paired.comparator == "cart_strong")
        ].set_index("method")
        for i, method in enumerate(MAIN_METHODS):
            if method not in case.index:
                continue
            row = case.loc[method]
            axis.errorbar(
                100 * row.difference,
                i,
                xerr=[
                    [100 * (row.difference - row.low)],
                    [100 * (row.high - row.difference)],
                ],
                fmt="o",
                color="#32658a",
                capsize=2,
                markersize=4,
            )
        axis.axvline(0, color="#888888", lw=1)
        axis.set_title(f"{spec['family'].title()}\nn = {spec['n']}")
        axis.set_yticks(range(len(MAIN_METHODS)), [LABELS[m] for m in MAIN_METHODS])
        axis.grid(axis="x", alpha=0.15)
    axes[0, 0].invert_yaxis()
    fig.supxlabel(
        "Accuracy difference from CART with the wider tuning grid (percentage points)"
    )
    fig.suptitle(
        "Mechanism illustrations: paired means and 95% intervals across 20 datasets",
        fontsize=13,
    )
    save_figure(fig, output, "accuracy")


def size_figure(curves, output):
    plt = plotting()
    specs = [illustrative_specs()[1], illustrative_specs()[2], illustrative_specs()[5]]
    methods = [
        "cart_strong",
        "holdout_selected",
        "holdout_common",
        "relative_bootstrap",
        "relative_subsample",
        "weighted_bootstrap",
        "aloof",
        "node_vote",
        "lookahead",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout="constrained")
    colors = plt.get_cmap("tab10").colors
    for axis, spec in zip(axes, specs):
        case = curves[curves.scenario == scenario_id(spec)]
        case = case.groupby(["method", "depth", "seed"])[["accuracy", "leaves"]].mean()
        means = case.groupby(["method", "depth"]).mean()
        for i, method in enumerate(methods):
            if method not in means.index.get_level_values("method"):
                continue
            rows = means.loc[method].sort_index()
            axis.plot(
                rows.leaves,
                100 * rows.accuracy,
                "o-",
                color=colors[i],
                label=LABELS[method],
                ms=4,
            )
        axis.set_title(f"{spec['family'].title()}, n = {spec['n']}")
        axis.set(xlabel="Mean number of leaves", ylabel="Test accuracy (%)")
        lower, upper = axis.get_ylim()
        axis.set_ylim(max(0, lower), min(100.5, upper))
        axis.grid(alpha=0.15)
    handles, labels = {}, {}
    for axis in axes:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            handles[label] = handle
            labels[label] = label
    fig.legend(
        handles.values(),
        labels.values(),
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    save_figure(fig, output, "size")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/study"))
    args = parser.parse_args()
    complete = json.loads((args.input / "complete.json").read_text())
    assert complete["complete"]
    summarize(args.input)
    metrics = pd.read_csv(args.input / "metrics.csv")
    paired = comparisons(metrics)
    paired.to_csv(args.input / "paired.csv", index=False)
    values = per_dataset(metrics)
    summary = values.groupby(["scenario", "method"])[["accuracy", "leaves"]].agg(
        ["mean", "std", "min", "max"]
    )
    summary.to_csv(args.input / "summary.csv")
    checks, eligible = screen(metrics)
    checks.to_csv(args.input / "screen.csv", index=False)
    noise_roots(args.input).to_csv(args.input / "noise_roots.csv", index=False)
    (args.input / "confirmation_tasks.json").write_text(
        json.dumps(confirmation_design(eligible), indent=2) + "\n"
    )
    figures = args.input / "figures"
    figures.mkdir(exist_ok=True)
    style()
    objective_figure(figures)
    accuracy_figure(paired, figures)
    size_figure(pd.read_csv(args.input / "curves.csv"), figures)
    print(
        f"{len(metrics)} fitted-tree/stump results; "
        f"{len(eligible)} method-condition pairs pass the screen."
    )


if __name__ == "__main__":
    main()
