"""Select one tree by fold-local validation of CART and grouped proposals."""

import argparse
import hashlib
import json
import os
import platform
import tempfile
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from cart_study.ablation import (
    PartitionTree,
    accuracy,
    interval,
    proposal_splits,
    require_equal,
    require_same_tree,
    rng_for,
    structure,
)
from cart_study.learned import (
    GapMap,
    cart_fit,
    conditions,
    observations,
    partition_groups,
    priorities,
)
from cart_study.trees import leaf_count

PRIMARY = ["cart", "raw", "mixed", "fixed_raw", "fixed_mixed"]
ARMS = ["selected", *PRIMARY, "constant"]
FAMILIES = {"constant": 0, "cart": 1, "raw": 2, "mixed": 3}


def candidates():
    grid = [dict(family="constant", depth=0, leaf=1, alpha=0.0)]
    grid.extend(
        dict(family="cart", depth=d, leaf=m, alpha=a)
        for d, m, a in product([1, 2, 5], [5, 10, 20], [0.0, 0.01, 0.03, 0.1])
    )
    grid.extend(
        dict(family=f, depth=d, leaf=m, alpha=0.0)
        for f, d, m in product(["raw", "mixed"], [1, 2, 5], [5, 10, 20])
    )
    return grid


def array_hash(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def tree_hash(model):
    if isinstance(model, PartitionTree):
        return hashlib.sha256(repr(structure(model)).encode()).hexdigest()
    tree = model.tree_
    return array_hash(
        tree.feature,
        tree.threshold,
        tree.value,
        tree.n_node_samples,
        tree.children_left,
        tree.children_right,
    )


def fit_grid(X, y, grid, key, stage, order, checks):
    """Every learned quantity uses only the rows supplied to this function."""
    seed, condition, replicate = key
    n = len(X)
    samples = rng_for(seed, condition, replicate, 1000 + 10 * stage).integers(
        n, size=(20, n)
    )
    seeds = rng_for(seed, condition, replicate, 1001 + 10 * stage).integers(
        2**31, size=21
    )
    models = {}
    mappings = {m: GapMap(m).fit(X) for m in [5, 10, 20]}
    grouped = {}
    for depth, minimum in product([1, 2, 5], [5, 10, 20]):
        options = dict(max_depth=depth, min_samples_leaf=minimum)
        mapping = mappings[minimum]
        raw = [
            cart_fit(X[ids], y[ids], options, s, order)
            for ids, s in zip(samples, seeds)
        ]
        mixed = raw[:10] + [
            cart_fit(X[ids], y[ids], options, s, order, mapping)
            for ids, s in zip(samples[10:], seeds[10:20])
        ]
        for family, pool in [("raw", raw), ("mixed", mixed)]:
            splits = [proposal_splits(t, depth) for t in pool]
            partitions, _, votes = partition_groups(X, splits)
            tie = priorities(partitions, (*key, stage, depth, minimum))
            grouped[family, depth, minimum] = PartitionTree(
                partitions, votes, tie, **options
            ).fit(X, y)
        if not mapping.thresholds:
            require_same_tree(
                structure(grouped["raw", depth, minimum]),
                structure(grouped["mixed", depth, minimum]),
                checks,
            )
            checks["no_gap_identity"] += 1
    for i, spec in enumerate(grid):
        if spec["family"] == "constant":
            options = dict(max_depth=1, min_samples_split=n + 1)
            model = cart_fit(X, y, options, seeds[-1], order)
            require_equal(leaf_count(model), 1, checks, "constant_one_leaf")
        elif spec["family"] == "cart":
            options = dict(
                max_depth=spec["depth"],
                min_samples_leaf=spec["leaf"],
                ccp_alpha=spec["alpha"],
            )
            model = cart_fit(X, y, options, seeds[-1], order)
        else:
            model = grouped[spec["family"], spec["depth"], spec["leaf"]]
        models[i] = model
    return models, dict(
        training_hash=array_hash(X, y),
        training_size=n,
        gaps={str(m): dict(g.thresholds) for m, g in mappings.items()},
    )


def choose(grid, errors, leaves, cap, family=None, iteration=None):
    eligible = [
        i
        for i in (range(len(grid)) if iteration is None else iteration)
        if grid[i]["depth"] <= cap
        and (family is None or grid[i]["family"] in [family, "constant"])
    ]
    return min(
        eligible,
        key=lambda i: (
            int(errors[i].sum()),
            int(leaves[i].sum()),
            FAMILIES[grid[i]["family"]],
            grid[i]["depth"],
            -grid[i]["leaf"],
            -grid[i]["alpha"],
            i,
        ),
    )


def train(X, y, config, condition, replicate):
    grid = config["candidates"]
    key = (config["seed"], condition, replicate)
    rng = rng_for(*key, 200)
    order = rng.permutation(X.shape[1])
    splitter = KFold(n_splits=5, shuffle=True, random_state=int(rng.integers(2**31)))
    folds = list(splitter.split(X))
    errors = np.zeros((len(grid), 5), dtype=int)
    leaves = np.zeros_like(errors)
    oof = np.zeros((len(grid), len(X)), dtype=np.int8)
    seen = np.zeros(len(X), dtype=int)
    fitted, audit, checks = [], [], Counter()
    for fold, (training, validation) in enumerate(folds):
        require_equal(
            np.intersect1d(training, validation).size, 0, checks, "fold_disjoint"
        )
        require_equal(
            np.sort(np.concatenate((training, validation))),
            np.arange(len(X)),
            checks,
            "fold_exhaustive",
        )
        seen[validation] += 1
        models, metadata = fit_grid(
            X[training], y[training], grid, key, fold, order, checks
        )
        require_equal(
            metadata["training_hash"],
            array_hash(X[training], y[training]),
            checks,
            "fold_training_inputs",
        )
        metadata["validation_rows"] = validation.tolist()
        audit.append(metadata)
        fitted.append(models)
        for i, model in models.items():
            prediction = model.predict(X[validation])
            oof[i, validation] = prediction
            errors[i, fold] = np.count_nonzero(prediction != y[validation])
            leaves[i, fold] = leaf_count(model)
    require_equal(seen, np.ones(len(X), dtype=int), checks, "one_validation_per_row")
    require_equal(
        np.count_nonzero(oof != y[None, :], axis=1),
        errors.sum(axis=1),
        checks,
        "oof_loss_alignment",
    )
    decisions = {}
    for cap in config["caps"]:
        selected = {"selected": choose(grid, errors, leaves, cap)}
        selected.update(
            {f: choose(grid, errors, leaves, cap, f) for f in ["cart", "raw", "mixed"]}
        )
        selected.update(
            {
                f"fixed_{f}": next(
                    i
                    for i, s in enumerate(grid)
                    if s["family"] == f and s["depth"] == cap and s["leaf"] == 10
                )
                for f in ["raw", "mixed"]
            }
        )
        selected["constant"] = 0
        require_equal(
            selected["selected"],
            choose(grid, errors, leaves, cap, iteration=reversed(range(len(grid)))),
            checks,
            "candidate_order",
        )
        if any(
            errors[selected["selected"]].sum() > errors[selected[f]].sum()
            for f in ["cart", "raw", "mixed"]
        ):
            raise AssertionError("Selection increased validation error")
        checks["nested_validation_minimum"] += 1
        if selected["selected"] not in [selected[f] for f in ["cart", "raw", "mixed"]]:
            raise AssertionError("Global winner differs from every family winner")
        checks["shared_family_winner"] += 1
        decisions[cap] = selected
    full, metadata = fit_grid(X, y, grid, key, 5, order, checks)
    audit.append(metadata)
    return dict(
        decisions=decisions,
        full=full,
        folds=fitted,
        errors=errors,
        leaves=leaves,
        oof=oof,
        audit=audit,
        checks=checks,
    )


def verify_protocol(config):
    """Make held-out rows disrupt a gap; their values cannot change that fold's fit."""
    rng = np.random.default_rng(9173)
    X = rng.normal(size=(100, 2))
    y = np.arange(100) % 2
    X[:, 0] = y + rng.uniform(-0.1, 0.1, 100)
    key = (config["seed"], 0, 9173)
    split_rng = rng_for(*key, 200)
    split_rng.permutation(2)
    splitter = KFold(5, shuffle=True, random_state=int(split_rng.integers(2**31)))
    _, validation = next(splitter.split(X))
    X[validation, 0] = np.where(np.arange(len(validation)) % 2, -10, 10)
    with threadpool_limits(limits=1):
        before = train(X, y, config, 0, 9173)
        changed_X, changed_y = X.copy(), y.copy()
        changed_X[validation] *= 100
        changed_y[validation] = 1 - changed_y[validation]
        after = train(changed_X, changed_y, config, 0, 9173)
    if 0 not in before["audit"][0]["gaps"]["10"]:
        raise AssertionError("Fold-local gap was not learned")
    if 0 in before["audit"][5]["gaps"]["10"]:
        raise AssertionError("Adversarial full-sample gap was not removed")
    checks = Counter(fold_local_gap=1)
    for i in range(len(config["candidates"])):
        require_equal(
            tree_hash(before["folds"][0][i]),
            tree_hash(after["folds"][0][i]),
            checks,
            "heldout_values_do_not_change_fold_fit",
        )
    return dict(checks)


def run_unit(config, condition, replicate):
    spec = config["conditions"][condition]
    X, truth, flips = observations(
        spec, spec["n"], config["seed"], condition, replicate, 0
    )
    y = truth ^ (flips < spec["eta"])
    with threadpool_limits(limits=1):
        fitted = train(X, y, config, condition, replicate)
    # Evaluation observations do not exist until selection and refitting finish.
    Xt, truth_test, flips_test = observations(
        spec, config["test_size"], config["seed"], condition, replicate, 100
    )
    yt = truth_test ^ (flips_test < spec["eta"])
    checks = fitted["checks"]
    needed = {i for choices in fitted["decisions"].values() for i in choices.values()}
    predictions = {i: fitted["full"][i].predict(Xt) for i in needed}
    risks = {
        i: accuracy(pred, truth_test, spec["eta"]) for i, pred in predictions.items()
    }
    require_equal(
        predictions[0],
        fitted["full"][0].predict(np.zeros_like(Xt)),
        checks,
        "constant_X",
    )
    fold_needed = {
        i
        for choices in fitted["decisions"].values()
        for arm, i in choices.items()
        if arm in ["selected", "cart", "raw", "mixed"]
    }
    fold_accuracy = {
        i: float(
            np.mean(
                [
                    accuracy(f[i].predict(Xt), truth_test, spec["eta"])
                    for f in fitted["folds"]
                ]
            )
        )
        for i in fold_needed
    }
    outcomes, contrasts, diagnostics = [], [], []
    for cap, choices in fitted["decisions"].items():
        picked, base = choices["selected"], choices["cart"]
        for arm, i in choices.items():
            model = fitted["full"][i]
            if spec["eta"] == 0.5:
                require_equal(risks[i], 0.5, checks, "null_risk")
            outcomes.append(
                dict(
                    cap=cap,
                    arm=arm,
                    candidate=i,
                    family=config["candidates"][i]["family"],
                    tree_hash=tree_hash(model),
                    accuracy=risks[i],
                    observed_accuracy=float(np.mean(predictions[i] == yt)),
                    leaves=leaf_count(model),
                    cv_accuracy=float(1 - fitted["errors"][i].sum() / len(X)),
                )
            )
        for arm in PRIMARY:
            other = choices[arm]
            contrasts.append(
                dict(cap=cap, contrast=arm, effect=risks[picked] - risks[other])
            )
        if picked == base:
            require_equal(
                predictions[picked], predictions[base], checks, "cart_fallback"
            )
        best_test = max(risks[choices[f]] for f in ["cart", "raw", "mixed"])
        if risks[picked] > best_test + 1e-12:
            raise AssertionError("Selected tree is not a restricted-family winner")
        gap_cv = (fitted["errors"][base].sum() - fitted["errors"][picked].sum()) / len(
            X
        )
        gap_fold = fold_accuracy[picked] - fold_accuracy[base]
        gap_full = risks[picked] - risks[base]
        diagnostics.append(
            dict(
                cap=cap,
                chosen_family=config["candidates"][picked]["family"],
                switched=int(picked != base),
                gain_vs_cart=gap_full,
                helped=max(0.0, gap_full),
                harmed=max(0.0, -gap_full),
                harmful_switch=int(gap_full < -1e-12),
                best_observed_family_accuracy=best_test,
                gap_to_best_observed=best_test - risks[picked],
                cv_gain_vs_cart=float(gap_cv),
                fold_test_gain_vs_cart=gap_fold,
                refit_gain_change=gap_full - gap_fold,
                fold_to_full_reversal=int(gap_fold * gap_full < -1e-12),
                oof_disagreement=float(
                    np.mean(fitted["oof"][picked] != fitted["oof"][base])
                ),
            )
        )
    return dict(
        condition=condition,
        replicate=replicate,
        spec=spec,
        input_hash=array_hash(X, y, Xt, truth_test),
        outcomes=outcomes,
        contrasts=contrasts,
        diagnostics=diagnostics,
        cv_errors=fitted["errors"].tolist(),
        cv_leaves=fitted["leaves"].tolist(),
        fitting=fitted["audit"],
        checks=dict(checks),
    )


def summarize(output, config):
    units = [
        json.loads((output / "units" / f"{c:02}_{r:04}.json").read_text())
        for c in range(len(config["conditions"]))
        for r in range(config["repetitions"])
    ]
    frames = {
        name: pd.DataFrame(
            dict(condition=u["condition"], replicate=u["replicate"], **u["spec"], **row)
            for u in units
            for row in u[name]
        )
        for name in ["outcomes", "contrasts", "diagnostics"]
    }
    keys = ["condition", "context", "n", "eta", "delta", "cap"]
    outcomes = frames["outcomes"]
    if (
        len(outcomes) != len(units) * len(config["caps"]) * len(ARMS)
        or outcomes.duplicated(["condition", "replicate", "cap", "arm"]).any()
    ):
        raise AssertionError("Incomplete or duplicate model evaluations")
    outcomes.groupby(keys + ["arm"]).agg(
        accuracy=("accuracy", "mean"),
        mcse=("accuracy", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        observed_accuracy=("observed_accuracy", "mean"),
        leaves=("leaves", "mean"),
        cv_accuracy=("cv_accuracy", "mean"),
    ).reset_index().to_csv(output / "summary.csv", index=False)
    family = len(config["conditions"]) * len(config["caps"]) * len(PRIMARY)
    estimates = []
    for names, group in frames["contrasts"].groupby(keys + ["contrast"]):
        if len(group) != config["repetitions"]:
            raise AssertionError("Incomplete paired contrasts")
        row = dict(zip(keys + ["contrast"], names))
        row.update(interval(100 * group.effect))
        simultaneous = interval(100 * group.effect, family)
        row.update(
            repetitions=len(group),
            simultaneous_lower=simultaneous["lower"],
            simultaneous_upper=simultaneous["upper"],
        )
        estimates.append(row)
    contrasts = pd.DataFrame(estimates)
    contrasts.to_csv(output / "contrasts.csv", index=False)
    outcomes.groupby(keys + ["arm", "family"]).size().div(config["repetitions"]).rename(
        "proportion"
    ).reset_index().to_csv(output / "choices.csv", index=False)
    diagnostic = frames["diagnostics"]
    diagnostic.groupby(keys).mean(numeric_only=True).drop(
        columns="replicate"
    ).reset_index().to_csv(output / "diagnostics.csv", index=False)
    checks = sum((Counter(u["checks"]) for u in units), Counter())
    checks.update(json.loads((output / "protocol.json").read_text()))
    (output / "invariances.json").write_text(json.dumps(dict(checks), indent=2) + "\n")
    plot(contrasts, output)
    print(
        f"Completed {len(units)} paired replicates; {family} primary contrasts.",
        flush=True,
    )


def plot(frame, output):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cart-mpl"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subset = frame[frame.contrast == "cart"]
    specs = subset.drop_duplicates("condition")
    labels = [
        f"{r.context.replace('_', ' ')}; n={r.n}, flips={r.eta:g}, jitter={r.delta:g}"
        for r in specs.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(10, 11))
    for cap, shift, color in [(2, -0.14, "#23678a"), (5, 0.14, "#a04b30")]:
        rows = subset[subset.cap == cap].set_index("condition").loc[specs.condition]
        ax.errorbar(
            rows["mean"],
            np.arange(len(rows)) + shift,
            xerr=[
                rows["mean"] - rows.simultaneous_lower,
                rows.simultaneous_upper - rows["mean"],
            ],
            fmt="o",
            capsize=2,
            color=color,
            label=f"Depth cap {cap}",
        )
    ax.axvline(0, color="#777777", linewidth=0.8)
    ax.axvline(-1, color="#777777", linewidth=0.8, linestyle=":")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Selected learner minus tuned CART (percentage points)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"accuracy.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def compute_unit(output, config, condition, replicate):
    path = output / "units" / f"{condition:02}_{replicate:04}.json"
    if path.exists():
        return
    result = run_unit(config, condition, replicate)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n"
    )
    temporary.replace(path)


def main():
    from cart_study.selection import compute_unit

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/selection"))
    parser.add_argument("--seed", type=int, default=20261011)
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
        caps=[2, 5],
        candidates=candidates(),
        primary=PRIMARY,
        folds=(
            "Five shuffled KFold folds independent of labels; "
            "shared across candidates"
        ),
        fitting=(
            "Gap maps, partitions, pools and leaf labels learned "
            "only from each fitting subset"
        ),
        selection=(
            "Minimum total out-of-fold mistakes; ties by summed fold leaves, "
            "constant/cart/raw/mixed, depth, descending leaf and alpha, index"
        ),
        refit=(
            "Selected configuration refit on all training rows "
            "before evaluation data generated"
        ),
        baselines=(
            "Same-fold family selection with shared constant; "
            "fixed raw/mixed at depth cap and leaf10; learned constant"
        ),
        inference=(
            "Paired-replicate t intervals; Bonferroni over all 340 primary contrasts"
        ),
        material_harm_pp=1.0,
        success=(
            "At least one positive simultaneous selector-CART contrast and every "
            "nonnull selector-CART lower bound above -1pp"
        ),
        diagnostics=(
            "Best observed test score among three family winners is optimistic, "
            "infeasible, and never used for selection; fold/full test "
            "differences descriptive"
        ),
        prior_information=(
            "Learned-study outcomes informed design; new seed, same 34 populations; "
            "smoke is engineering only, no outcome-driven tuning"
        ),
        missing=(
            "Any failed fit/check aborts; no dropped units; exact source/config resume"
        ),
        versions=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            sklearn=sklearn.__version__,
        ),
        source_hashes={
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in [
                "selection.py",
                "learned.py",
                "ablation.py",
                "trees.py",
                "baselines.py",
            ]
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
    (args.output / "protocol.json").write_text(
        json.dumps(verify_protocol(config), indent=2) + "\n"
    )
    (args.output / "units").mkdir(exist_ok=True)
    Parallel(n_jobs=args.jobs, verbose=10)(
        delayed(compute_unit)(args.output, config, c, r)
        for c in range(len(config["conditions"]))
        for r in range(config["repetitions"])
    )
    summarize(args.output, config)


if __name__ == "__main__":
    main()
