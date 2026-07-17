"""Editable training artifact — the ONLY file the Phase 1 loop modifies.

The orchestrator's patcher rewrites exactly one value inside the marker block
below per experiment. Everything else is expected to stay put in Phase 1
(later phases may allow broader edits within src/**).

Stdlib-only and deterministic: fixed data seed (via evaluation.dataset) and a
fixed shuffle seed make (hyperparams -> score) a pure function. Squares are
computed with `d * d` rather than `d ** 2` so that divergence overflows to
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
    "feature_scaling": False,
}
# --- HYPERPARAMS-END ---

SHUFFLE_SEED = 1337


def standardize_stats(xs: list[list[float]]) -> tuple[list[float], list[float]]:
    n = len(xs)
    means = [sum(row[j] for row in xs) / n for j in range(N_FEATURES)]
    stds = []
    for j in range(N_FEATURES):
        var = sum((row[j] - means[j]) * (row[j] - means[j]) for row in xs) / n
        stds.append(math.sqrt(var) if var > 0.0 else 1.0)
    return means, stds


def apply_scaling(
    xs: list[list[float]], means: list[float], stds: list[float]
) -> list[list[float]]:
    return [
        [(row[j] - means[j]) / stds[j] for j in range(N_FEATURES)] for row in xs
    ]


def _rmse(w: list[float], b: float, xs: list[list[float]], ys: list[float]) -> float:
    sse = 0.0
    for i in range(len(xs)):
        pred = b + sum(w[j] * xs[i][j] for j in range(N_FEATURES))
        d = pred - ys[i]
        sse += d * d
    return math.sqrt(sse / len(xs))


def train() -> dict:
    started = time.perf_counter()
    hp = dict(HYPERPARAMS)

    epochs = int(hp["epochs"])
    if os.environ.get("AUTORESEARCH_SMOKE") == "1":
        epochs = min(epochs, 2)

    xs, ys = load_train()

    feature_means: list[float] | None = None
    feature_stds: list[float] | None = None
    if hp["feature_scaling"]:
        feature_means, feature_stds = standardize_stats(xs)
        xs = apply_scaling(xs, feature_means, feature_stds)

    lr = float(hp["lr"])
    beta = float(hp["momentum"])
    l2 = float(hp["l2"])
    batch_size = max(1, int(hp["batch_size"]))

    w = [0.0] * N_FEATURES
    b = 0.0
    vw = [0.0] * N_FEATURES
    vb = 0.0

    # Single shuffle RNG reused across epochs: epoch k's permutation depends
    # only on (SHUFFLE_SEED, k), never on wall clock or hash randomization.
    rng = random.Random(SHUFFLE_SEED)
    indices = list(range(len(xs)))

    for _ in range(epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            gw = [0.0] * N_FEATURES
            gb = 0.0
            for i in batch:
                pred = b + sum(w[j] * xs[i][j] for j in range(N_FEATURES))
                err = pred - ys[i]
                for j in range(N_FEATURES):
                    gw[j] += err * xs[i][j]
                gb += err
            # Factor-2 MSE gradient convention (pinned: the documented
            # divergence thresholds depend on this factor).
            inv = 2.0 / len(batch)
            for j in range(N_FEATURES):
                g = gw[j] * inv + 2.0 * l2 * w[j]
                vw[j] = beta * vw[j] - lr * g
                w[j] += vw[j]
            vb = beta * vb - lr * (gb * inv)
            b += vb

    train_rmse = _rmse(w, b, xs, ys)

    return {
        "schema_version": 1,
        "weights": w,
        "bias": b,
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
