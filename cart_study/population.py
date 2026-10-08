"""Evaluate split-frequency reconstruction in the population XOR model."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def support_probabilities(p):
    if p < 3:
        raise ValueError("The model requires at least three binary features")
    a = 1 - (1 - 1 / (p - 1)) ** 2
    return 2 / p + (p - 2) * a / p, 1 / p + (p - 3) * a / p


def reconstruction_accuracy(counts):
    """Exact population accuracy, averaged over uniform ties in the final tree."""
    counts = np.asarray(counts)
    top = counts == counts.max(axis=1, keepdims=True)
    recovery = np.zeros(len(counts))
    for signal in (0, 1):
        remaining = counts.copy()
        remaining[:, signal] = -1
        child_top = remaining == remaining.max(axis=1, keepdims=True)
        recovery += (
            top[:, signal]
            / top.sum(axis=1)
            * child_top[:, 1 - signal]
            / child_top.sum(axis=1)
        )
    return 0.5 + 0.5 * recovery


def add_trees(counts, number, rng):
    """Draw population Gini trees; count each feature at most once per tree."""
    repetitions, p = counts.shape
    for start in range(0, repetitions, 64):
        end = min(start + 64, repetitions)
        root = rng.integers(p, size=(end - start, number))
        left = rng.integers(p - 1, size=root.shape)
        right = rng.integers(p - 1, size=root.shape)
        left += left >= root
        right += right >= root
        left = np.where(root < 2, 1 - root, left)
        right = np.where(root < 2, 1 - root, right)
        rows = np.broadcast_to(np.arange(start, end)[:, None], root.shape)
        np.add.at(counts, (rows, root), 1)
        np.add.at(counts, (rows, left), 1)
        distinct = right != left
        np.add.at(counts, (rows[distinct], right[distinct]), 1)


def simulate(features, sizes, repetitions, seed):
    rows = []
    for p in features:
        q_signal, q_noise = support_probabilities(p)
        gap = q_signal - q_noise
        rng = np.random.default_rng(np.random.SeedSequence([seed, p]))
        counts = np.zeros((repetitions, p), dtype=np.int64)
        previous = 0
        for size in sorted(set(sizes)):
            add_trees(counts, size - previous, rng)
            accuracy = reconstruction_accuracy(counts)
            rows.append(
                dict(
                    features=p,
                    pool_size=size,
                    repetitions=repetitions,
                    accuracy=accuracy.mean(),
                    monte_carlo_se=accuracy.std(ddof=1) / np.sqrt(repetitions),
                    cart_accuracy=0.5 + 1 / p,
                    lookahead_accuracy=1.0,
                    signal_support=q_signal,
                    noise_support=q_noise,
                    support_gap=gap,
                    mean_signal_support=(counts[:, :2] / size).mean(),
                    mean_noise_support=(counts[:, 2:] / size).mean(),
                    risk_upper_bound=min(0.5, (p - 2) * np.exp(-size * gap**2 / 2)),
                )
            )
            previous = size
    return pd.DataFrame(rows)


def figure(frame, output):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cart-mpl"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    features = sorted(frame.features.unique())
    fig, axes = plt.subplots(
        1, len(features), figsize=(3.2 * len(features), 3.3), sharey=True, squeeze=False
    )
    for ax, p in zip(axes[0], features):
        rows = frame[frame.features == p].sort_values("pool_size")
        ax.plot(rows.pool_size, 100 * rows.accuracy, color="#23678a", marker="o")
        ax.fill_between(
            rows.pool_size,
            100 * (rows.accuracy - 1.96 * rows.monte_carlo_se).clip(0, 1),
            100 * (rows.accuracy + 1.96 * rows.monte_carlo_se).clip(0, 1),
            color="#23678a",
            alpha=0.2,
        )
        ax.axhline(100 * (0.5 + 1 / p), color="#a04b30", linestyle="--")
        ax.axhline(100, color="#555555", linestyle=":")
        ax.set(xscale="log", xlabel="Pool trees", title=f"{p} binary features")
        ax.set_ylim(48, 103)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Population accuracy (%)")
    axes[0, -1].plot([], [], color="#23678a", label="Split frequency")
    axes[0, -1].plot([], [], color="#a04b30", linestyle="--", label="Greedy CART")
    axes[0, -1].plot(
        [], [], color="#555555", linestyle=":", label="Two-level lookahead"
    )
    fig.legend(
        *axes[0, -1].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"accuracy.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/population"))
    parser.add_argument("--features", type=int, nargs="+", default=[3, 5, 10, 20])
    parser.add_argument(
        "--pool-sizes", type=int, nargs="+", default=[1, 5, 20, 100, 500, 2000, 10000]
    )
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.repetitions < 2 or min(args.pool_sizes) < 1 or min(args.features) < 3:
        parser.error("Use at least two repetitions, positive pool sizes, and p >= 3")
    frame = simulate(args.features, args.pool_sizes, args.repetitions, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "summary.csv", index=False)
    config = dict(
        features=args.features,
        pool_sizes=args.pool_sizes,
        repetitions=args.repetitions,
        seed=args.seed,
        population="Independent balanced bits; Y = X1 XOR X2; exact population Gini",
        pool="Depth two; zero-gain splits allowed; independent uniform node ties",
        reconstruction="Depth two; global split counts; no cutoff; uniform node ties",
        evaluation="Exact population error averaged over reconstruction ties",
        source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    figure(frame, args.output)
    print(
        frame[["features", "pool_size", "accuracy", "monte_carlo_se"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
