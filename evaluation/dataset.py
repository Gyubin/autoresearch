"""Euclidean-TSP instance generator for the Phase 6c research domain.

PROTECTED FILE (listed in research_contract.yaml protected_globs).

The task is now combinatorial optimization: the candidate solver in src/train.py
is handed problem INSTANCES (city coordinates) and must return a tour (a
permutation); the trusted evaluator recomputes the tour length itself.

Split hygiene (unchanged model, new payload):
  * The training instances use a public seed (TRAIN_SEED) and are available to
    the solver via load_train() for offline development.
  * Three hidden held-out instance sets (dev / gate / test) have independent
    seeds in evaluation/heldout_config.json, generated at `orchestrator.py init`
    and NOT tracked by git, so candidate worktrees physically lack the seeds.
    Only the root evaluator calls load_split(). dev drives keep/reject search,
    gate is the blind admission split, test stays untouched until the report.

Seed non-leakage (Phase 6c CRITICAL invariant):
  * The evaluator hands the solver only the INSTANCE COORDINATES, never the seed.
    Instance ids are OPAQUE indices ("i0", "i1", …) — the seed integer never
    appears in any structure that crosses into the sandbox, so a solver cannot
    regenerate other splits or the wider distribution. `fingerprint()` proves
    which instances scored a run without leaking the seed.

Determinism:
  * Coordinates are integers on a GRID lattice via rng.random() (guaranteed
    stable across Python versions, unlike randint/shuffle). Euclidean distances
    use the TSPLIB EUC_2D integer rounding nint(d) = int(d + 0.5), so tour
    lengths are integer and byte-stable on a machine; fingerprint() catches any
    cross-machine libm drift.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

# Number of cities per instance. Cross-checked against evaluation/evaluate.py's
# hardcoded N_CITIES at `orchestrator.py init` (fails fast on drift).
N_CITIES = 60
GRID = 1_000_000  # integer coordinate lattice; keeps distances byte-stable

TRAIN_SEED = 20260401  # public
# Public development instances the solver/coder iterates on offline. Wider than a
# handful so tuning a solver against them does not trivially overfit before the
# hidden dev split is ever seen.
N_TRAIN_INSTANCES = 40

# Instances per split (PUBLIC constants; the seeds are the secret). test is sized
# so the Phase 5 paired bootstrap keeps real resolution: 160 instances x
# finalist_seeds pairs. A size read from the untracked config could be gamed, so
# sizes stay hardcoded here.
SPLIT_SIZES = {"dev": 40, "gate": 40, "test": 160}

HELDOUT_SPLITS = ("dev", "gate", "test")


def euclid_nint(a: list[int], b: list[int]) -> int:
    """TSPLIB EUC_2D integer distance nint(sqrt(dx^2 + dy^2))."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return int(math.sqrt(dx * dx + dy * dy) + 0.5)


def _instance(rng: random.Random, index: int) -> dict:
    # OPAQUE id — a bare index, NEVER the seed (Phase 6c seed-non-leakage).
    coords = [[int(rng.random() * GRID), int(rng.random() * GRID)]
              for _ in range(N_CITIES)]
    return {"instance_id": f"i{index}", "coords": coords}


def generate(seed: int, size: int) -> list[dict]:
    rng = random.Random(seed)
    return [_instance(rng, k) for k in range(size)]


def load_train() -> list[dict]:
    """Public training instances. Available to candidate solver code."""
    return generate(TRAIN_SEED, N_TRAIN_INSTANCES)


def _split_seeds(entry: dict, split: str) -> list[int]:
    """Seed list for a split entry (config schema v4).

    dev/gate carry a single ``seed`` (a 1-element list here); ``test`` carries a
    ``seeds`` list of N pairwise-distinct hidden seeds for Phase 5 multi-seed
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
) -> list[dict]:
    """One hidden held-out instance set (dev/gate/test). Root evaluator only.

    Config schema v4: {"schema_version": 4, "splits": {
        "dev": {"seed": int}, "gate": {"seed": int},
        "test": {"seeds": [int, ...]}}}.

    ``seed_index`` selects among the test split's seeds (Phase 5 multi-seed
    finalist reproduction); dev/gate accept only seed_index 0.
    """
    if split not in SPLIT_SIZES:
        raise KeyError(f"unknown split {split!r} (have: {sorted(SPLIT_SIZES)})")
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if cfg.get("schema_version") != 4:
        raise KeyError(
            f"heldout config schema_version must be 4, got "
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


def flat_coords(instances: list[dict]) -> list[int]:
    """Flattened integer coordinate stream (for fingerprinting)."""
    out: list[int] = []
    for inst in instances:
        for xy in inst["coords"]:
            out.append(xy[0])
            out.append(xy[1])
    return out


def fingerprint(values: list[int], k: int = 64) -> str:
    """Hash of the first k coordinate values.

    Proves which instances scored a run (and detects cross-machine libm drift in
    downstream distance math) without leaking the seed itself.
    """
    payload = ",".join(repr(v) for v in values[:k]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
