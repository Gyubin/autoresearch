"""Synthetic regression dataset for the Phase 1 mock task.

PROTECTED FILE (listed in research_contract.yaml protected_globs).

Split hygiene:
  * The training split uses a public seed (TRAIN_SEED) and is available to
    candidate code via load_train().
  * Three hidden held-out splits (dev / gate / test) have independent seeds
    in evaluation/heldout_config.json, which is generated at
    `orchestrator.py init` time and deliberately NOT tracked by git —
    worktrees materialize tracked files only, so candidate workspaces
    physically lack the seeds. Only the root evaluator calls load_split().
    dev drives keep/reject search, gate is the blind admission split, test
    stays untouched until the campaign-end report.

Determinism:
  * Gaussian samples come from a hand-rolled Box–Muller transform over
    rng.random(), because random.random() is guaranteed stable across Python
    versions while distribution helpers such as rng.gauss() are not.
  * All RNGs are instance-scoped; nothing touches the module-level random
    state.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

N_FEATURES = 8

# Heterogeneous feature scales make feature_scaling a real intervention and
# couple it with the usable learning-rate range (condition number ~625 when
# unscaled). The interaction term sits on x0*x1 — both scale 1.0 — so the
# irreducible floor of a linear model is unaffected by the scales.
SCALES = [1.0, 1.0, 5.0, 0.2, 1.0, 3.0, 1.0, 0.5]
TRUE_W = [0.8, -0.5, 0.15, 2.0, -0.7, 0.2, -0.2, 1.2]
TRUE_BIAS = 0.3
INTERACTION = 0.3  # coefficient on x0*x1; a linear model cannot capture it
NOISE_STD = 0.25

TRAIN_SEED = 20260401  # public
N_TRAIN = 600
# Split sizes are PUBLIC constants hardcoded here (the seeds are the secret):
# a size read from the untracked config could be manipulated (e.g. a 1-row
# gate split), so the evaluator trusts only this file for sizes.
SPLIT_SIZES = {"dev": 400, "gate": 400, "test": 600}


def _gauss(rng: random.Random) -> float:
    """Standard normal via Box–Muller from rng.random() (version-stable)."""
    u1 = 1.0 - rng.random()  # (0, 1], avoids log(0)
    u2 = rng.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def generate(seed: int, size: int) -> tuple[list[list[float]], list[float]]:
    rng = random.Random(seed)
    xs: list[list[float]] = []
    ys: list[float] = []
    for _ in range(size):
        x = [_gauss(rng) * SCALES[j] for j in range(N_FEATURES)]
        y = TRUE_BIAS + sum(w * v for w, v in zip(TRUE_W, x))
        y += INTERACTION * x[0] * x[1]
        y += NOISE_STD * _gauss(rng)
        xs.append(x)
        ys.append(y)
    return xs, ys


def load_train() -> tuple[list[list[float]], list[float]]:
    """Public training split. Available to candidate code."""
    return generate(TRAIN_SEED, N_TRAIN)


HELDOUT_SPLITS = ("dev", "gate", "test")


def _split_seeds(entry: dict, split: str) -> list[int]:
    """Seed list for a split entry (config schema v3).

    dev/gate carry a single ``seed`` (a 1-element list here); ``test`` carries
    a ``seeds`` list of N pairwise-distinct hidden seeds for Phase 5 multi-seed
    finalist reproduction. Exactly one of the two keys must be present.
    """
    has_seed = "seed" in entry
    has_seeds = "seeds" in entry
    if has_seed == has_seeds:
        raise KeyError(
            f"split {split!r} entry must have exactly one of 'seed'/'seeds'")
    if has_seed:
        return [int(entry["seed"])]
    seeds = entry["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise KeyError(f"split {split!r} 'seeds' must be a non-empty list")
    ints = [int(s) for s in seeds]
    if len(set(ints)) != len(ints):
        raise KeyError(f"split {split!r} 'seeds' must be pairwise-distinct")
    return ints


def load_split(
    config_path: str | Path, split: str, seed_index: int = 0
) -> tuple[list[list[float]], list[float]]:
    """One hidden held-out split (dev/gate/test). Root evaluator only.

    Config schema v3: {"schema_version": 3, "splits": {
        "dev": {"seed": int}, "gate": {"seed": int},
        "test": {"seeds": [int, ...]}}}.

    ``seed_index`` selects among the test split's seeds (Phase 5 multi-seed
    finalist reproduction); dev/gate accept only seed_index 0.
    """
    if split not in SPLIT_SIZES:
        raise KeyError(f"unknown split {split!r} (have: {sorted(SPLIT_SIZES)})")
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if cfg.get("schema_version") != 3:
        raise KeyError(
            f"heldout config schema_version must be 3, got "
            f"{cfg.get('schema_version')!r} (stale config? re-run "
            f"`orchestrator.py init --force`)")
    splits = cfg.get("splits")
    if not isinstance(splits, dict) or split not in splits:
        have = sorted(splits) if isinstance(splits, dict) else "none"
        raise KeyError(f"split {split!r} missing from heldout config (have: {have})")
    seeds = _split_seeds(splits[split], split)
    if not 0 <= seed_index < len(seeds):
        raise IndexError(
            f"seed_index {seed_index} out of range for split {split!r} "
            f"({len(seeds)} seed(s))")
    return generate(seeds[seed_index], SPLIT_SIZES[split])


def fingerprint(values: list[float], k: int = 32) -> str:
    """Hash of the first k targets.

    Proves which data scored a run (and detects cross-machine libm drift)
    without leaking the seed itself.
    """
    payload = ",".join(repr(v) for v in values[:k]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
