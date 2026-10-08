"""Population definitions and shared experiment settings."""

import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_wine,
    make_classification,
)

NODE_SYNTHETIC = ("low_noise", "moderate_noise", "heavy_noise", "very_heavy_noise")


REAL = ("breast_cancer", "wine", "digits")


CONSENSUS_SCENARIOS = (
    "linear_baseline",
    "many_noise_features",
    "small_sample",
    "duplicates_3",
    "high_noise_60",
    "duplicates_6",
    "duplicates_6_noise1",
    "imbalance_60_40",
    "sparse_highd",
    "xor",
)


TAUS = (0.0, 0.25, 0.35, 0.5)


SCALES = (0.3, 0.6, 1.0)


ALPHAS = (0.0, 0.001, 0.005, 0.01, 0.02)


def dataset(name, seed):
    if name in REAL:
        loader = dict(
            breast_cancer=load_breast_cancer, wine=load_wine, digits=load_digits
        )[name]
        return loader(return_X_y=True)
    if name in NODE_SYNTHETIC:
        informative, noise = {
            "low_noise": (7, 0.01),
            "moderate_noise": (5, 0.15),
            "heavy_noise": (3, 0.3),
            "very_heavy_noise": (2, 0.45),
        }[name]
        return make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=informative,
            n_redundant=2,
            flip_y=noise,
            random_state=seed,
        )
    rng = np.random.default_rng(seed)
    if name == "xor":
        X = rng.standard_normal((600, 4))
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
        return np.hstack([X, rng.standard_normal((600, 16))]), y
    if name.startswith("duplicates"):
        k, p = (3, 20) if name == "duplicates_3" else (6, 30)
        base = rng.standard_normal((600, k))
        duplicate = np.repeat(base, 2, axis=1) + rng.normal(0, 0.05, (600, 2 * k))
        X = np.hstack([duplicate, rng.standard_normal((600, p - 2 * k))])
        coef = rng.normal(size=k)
        noise = 0.3 if k == 3 else 0.35
        if name.endswith("noise1"):
            noise = 1.0
        latent = base @ coef + rng.normal(scale=noise, size=600)
    elif name == "sparse_highd":
        X = rng.standard_normal((800, 1000))
        columns = np.sort(rng.choice(1000, 10, replace=False))
        coef = rng.normal(size=10)
        latent = X[:, columns] @ coef + rng.normal(scale=0.5, size=800)
    else:
        n, p, k, noise = {
            "linear_baseline": (600, 12, 3, 0.3),
            "many_noise_features": (600, 50, 3, 0.3),
            "small_sample": (200, 12, 3, 0.3),
            "high_noise_60": (600, 60, 8, 0.45),
            "imbalance_60_40": (800, 40, 6, 0.3),
        }[name]
        X = rng.standard_normal((n, p))
        coef = rng.normal(size=k)
        latent = X[:, :k] @ coef + rng.normal(scale=noise, size=n)
    y = (latent > np.median(latent)).astype(int)
    if name == "imbalance_60_40":
        y[: int(0.2 * len(y))] = 0
    return X, y


def scaled_eta(scale, support, tau, B):
    n_candidates = sum(frequency >= tau for frequency in support.values())
    return float(scale * np.sqrt(2 * np.log(n_candidates + 1) / (B + 1)))


def objective_population(mixture):
    atoms = np.array([(a, b, y) for a in (0, 1) for b in (0, 1) for y in (0, 1)])
    original = np.array([0, 14, 0, 6, 14, 22, 36, 8]) / 100
    agreement = (atoms[:, 2] == 1 - atoms[:, 1]) / 4
    return atoms, (1 - mixture) * original + mixture * agreement


def scenarios():
    specs = []
    for n in (200, 800):
        for mixture in (0.0, 0.5, 1.0):
            specs.append(dict(family="objective", n=n, p=2, mixture=mixture))
        for noise in (0.1, 0.4):
            for p in (2, 20):
                specs.append(dict(family="noise", n=n, p=p, noise=noise))
        for mixture in (0.0, 0.25, 1.0):
            specs.append(dict(family="interaction", n=n, p=20, mixture=mixture))
    return specs


def scenario_id(spec):
    return "_".join(f"{key}-{value}" for key, value in sorted(spec.items()))


def generate(spec, n, seed):
    rng = np.random.default_rng(seed)
    family, p = spec["family"], spec["p"]
    if family == "objective":
        atoms, mass = objective_population(spec["mixture"])
        values = atoms[rng.choice(len(atoms), size=n, p=mass)]
        return values[:, :2].astype(float), values[:, 2]
    if family == "noise":
        X = rng.normal(size=(n, p))
        X[:, 0] = rng.integers(0, 2, n)
        y = X[:, 0].astype(int) ^ (rng.random(n) < spec["noise"])
        return X, y.astype(int)
    if family == "interaction":
        X = rng.integers(0, 2, size=(n, p))
        y = np.where(rng.random(n) < spec["mixture"], X[:, 0], X[:, 0] ^ X[:, 1])
        return X.astype(float), y
    raise ValueError(f"Unknown family {family}")
