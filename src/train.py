"""Editable training artifact — the primary surface the loop modifies.

Two kinds of intervention touch this file:
  * The deterministic patcher rewrites exactly one value inside the
    HYPERPARAMS marker block (hyperparameter experiments).
  * The LLM coding worker may edit anything under src/** (code experiments) —
    e.g. extending FEATURE_SPEC to add engineered inputs the linear readout
    cannot otherwise represent.

FEATURE_SPEC declares the model inputs as products of raw features: each term
is a list of raw-feature indices multiplied together. The identity spec
([[0], [1], ..., [7]]) is a plain linear model over the 8 raw features; adding
[0, 1] appends an x0*x1 interaction input. The evaluator applies the SAME spec
from artifacts/model.json when scoring, so train and eval never disagree.

Stdlib-only and deterministic: fixed data seed (via evaluation.dataset) and a
fixed shuffle seed make (hyperparams, code) -> score a pure function. Squares
are computed with `d * d` rather than `d ** 2` so that divergence overflows to
inf/nan (a scoreable degenerate result) instead of raising OverflowError.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.dataset import N_FEATURES, load_train

# --- HYPERPARAMS-BEGIN (auto-patched; do not edit by hand) ---
HYPERPARAMS = {
    "lr": 0.005,
    "epochs": 30,
    "momentum": 0.0,
    "l2": 0.0,
    "batch_size": 32,
    "feature_scaling": True,
}
# --- HYPERPARAMS-END ---

# Engineered-feature spec: each term is a list of raw-feature indices whose
# product forms one model input. The default is the identity (plain linear
# model over the 8 raw features). A coding experiment may extend this — e.g.
# append [0, 1] for an x0*x1 interaction term.
FEATURE_SPEC = [[j] for j in range(N_FEATURES)]

SHUFFLE_SEED = 1337


def engineer(x: list[float]) -> list[float]:
    out = []
    for term in FEATURE_SPEC:
        v = 1.0
        for i in term:
            v *= x[i]
        out.append(v)
    return out


def standardize_stats(xs: list[list[float]]) -> tuple[list[float], list[float]]:
    n = len(xs)
    dim = len(xs[0])
    means = [sum(row[j] for row in xs) / n for j in range(dim)]
    stds = []
    for j in range(dim):
        var = sum((row[j] - means[j]) * (row[j] - means[j]) for row in xs) / n
        stds.append(math.sqrt(var) if var > 0.0 else 1.0)
    return means, stds


def apply_scaling(
    xs: list[list[float]], means: list[float], stds: list[float]
) -> list[list[float]]:
    dim = len(means)
    return [[(row[j] - means[j]) / stds[j] for j in range(dim)] for row in xs]


def _rmse(w: list[float], b: float, xs: list[list[float]], ys: list[float]) -> float:
    dim = len(w)
    sse = 0.0
    for i in range(len(xs)):
        pred = b + sum(w[j] * xs[i][j] for j in range(dim))
        d = pred - ys[i]
        sse += d * d
    return math.sqrt(sse / len(xs))


def train() -> dict:
    started = time.perf_counter()
    hp = dict(HYPERPARAMS)

    epochs = int(hp["epochs"])
    if os.environ.get("AUTORESEARCH_SMOKE") == "1":
        epochs = min(epochs, 2)

    xs_raw, ys = load_train()
    xs = [engineer(row) for row in xs_raw]  # model inputs per FEATURE_SPEC
    dim = len(FEATURE_SPEC)

    feature_means: list[float] | None = None
    feature_stds: list[float] | None = None
    if hp["feature_scaling"]:
        feature_means, feature_stds = standardize_stats(xs)
        xs = apply_scaling(xs, feature_means, feature_stds)

    lr = float(hp["lr"])
    beta = float(hp["momentum"])
    l2 = float(hp["l2"])
    batch_size = max(1, int(hp["batch_size"]))

    w = [0.0] * dim
    b = 0.0
    vw = [0.0] * dim
    vb = 0.0

    # Single shuffle RNG reused across epochs: epoch k's permutation depends
    # only on (SHUFFLE_SEED, k), never on wall clock or hash randomization.
    rng = random.Random(SHUFFLE_SEED)
    indices = list(range(len(xs)))

    for _ in range(epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            gw = [0.0] * dim
            gb = 0.0
            for i in batch:
                pred = b + sum(w[j] * xs[i][j] for j in range(dim))
                err = pred - ys[i]
                for j in range(dim):
                    gw[j] += err * xs[i][j]
                gb += err
            # Factor-2 MSE gradient convention (pinned: the documented
            # divergence thresholds depend on this factor).
            inv = 2.0 / len(batch)
            for j in range(dim):
                g = gw[j] * inv + 2.0 * l2 * w[j]
                vw[j] = beta * vw[j] - lr * g
                w[j] += vw[j]
            vb = beta * vb - lr * (gb * inv)
            b += vb

    train_rmse = _rmse(w, b, xs, ys)

    return {
        "schema_version": 2,
        "weights": w,
        "bias": b,
        "feature_spec": FEATURE_SPEC,
        "feature_means": feature_means,
        "feature_stds": feature_stds,
        "hyperparams": hp,
        "train_rmse": train_rmse,
        "train_seconds": time.perf_counter() - started,
    }


def main() -> None:
    artifact = train()
    out_dir = ROOT / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model.json").write_text(
        json.dumps(artifact, sort_keys=True), encoding="utf-8"
    )
    print(
        f"train_rmse={artifact['train_rmse']!r} "
        f"train_seconds={artifact['train_seconds']:.3f}"
    )


if __name__ == "__main__":
    main()
