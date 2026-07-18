"""Editable solver artifact — the primary surface the loop modifies (Phase 6c).

The task is Euclidean TSP: given problem instances (city coordinates), produce a
tour (a permutation of city indices) that minimizes total Euclidean length. Two
kinds of intervention touch this file:
  * The deterministic patcher rewrites exactly one value inside the HYPERPARAMS
    marker block (search hyperparameters).
  * The LLM coding worker may edit anything under src/** — e.g. swap NEIGHBORHOOD
    (2-opt -> Or-opt), change the acceptance rule, add tabu memory, or add a
    greedy-edge construction. This is the "meaty" algorithmic surface.

Where the instances come from:
  * The trusted evaluator hands this solver the split's instances (coordinates
    ONLY, never the seed) via the AUTORESEARCH_INSTANCES file path. When unset
    (manual runs / smoke), it falls back to the PUBLIC training instances.
  * The solver emits artifacts/solution.json; the evaluator VALIDATES the
    permutation and RECOMPUTES the tour length itself — a self-reported length is
    ignored, so a solver cannot inflate its own score.

Stdlib-only and deterministic: a fixed SOLVER_SEED makes (instances, HYPERPARAMS,
code) -> solution a pure function. Distances use the same integer TSPLIB EUC_2D
rounding as the evaluator so the solver optimizes exactly what is scored.
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

from evaluation.dataset import N_CITIES, euclid_nint, load_train

# --- HYPERPARAMS-BEGIN (auto-patched; do not edit by hand) ---
HYPERPARAMS = {
    "use_nn_construction": True,
    "max_iterations": 20000,
    "restarts": 1,
    "initial_temperature": 0.0,
    "cooling_rate": 0.995,
    "segment_max": 3,
    "perturbation_strength": 4,
}
# --- HYPERPARAMS-END ---

# Code-level knob the LLM coder may edit (like FEATURE_SPEC in the prior domain):
# the local-search neighborhood. "two_opt" (segment reversal) or "or_opt"
# (relocate a short segment). Structural search changes are coder-only.
NEIGHBORHOOD = "two_opt"

SOLVER_SEED = 1337  # fixed; determinism (like the prior SHUFFLE_SEED)


def tour_length(tour: list[int], coords: list[list[int]]) -> int:
    n = len(tour)
    return sum(euclid_nint(coords[tour[i]], coords[tour[(i + 1) % n]])
               for i in range(n))


def _rand_int(rng: random.Random, lo: int, hi: int) -> int:
    """Inclusive [lo, hi] via rng.random() (version-stable, unlike randint)."""
    return lo + int(rng.random() * (hi - lo + 1))


def nearest_neighbor(coords: list[list[int]]) -> list[int]:
    n = len(coords)
    unvisited = set(range(1, n))
    tour = [0]
    while unvisited:
        last = tour[-1]
        nxt = min(unvisited, key=lambda c: euclid_nint(coords[last], coords[c]))
        tour.append(nxt)
        unvisited.discard(nxt)
    return tour


def _two_opt_move(tour: list[int], coords: list[list[int]],
                  rng: random.Random) -> tuple[int, int, int]:
    """Propose reversing tour[i..j]; return (i, j, delta) with O(1) delta."""
    n = len(tour)
    i = _rand_int(rng, 1, n - 2)
    j = _rand_int(rng, i + 1, n - 1)
    a, b = tour[i - 1], tour[i]
    c, d = tour[j], tour[(j + 1) % n]
    before = euclid_nint(coords[a], coords[b]) + euclid_nint(coords[c], coords[d])
    after = euclid_nint(coords[a], coords[c]) + euclid_nint(coords[b], coords[d])
    return i, j, after - before


def _or_opt_move(tour: list[int], coords: list[list[int]], rng: random.Random,
                 segment_max: int) -> tuple[list[int], int] | None:
    """Relocate a segment of length 1..segment_max to another position."""
    n = len(tour)
    seg_len = _rand_int(rng, 1, max(1, min(segment_max, n - 2)))
    i = _rand_int(rng, 0, n - seg_len)
    segment = tour[i:i + seg_len]
    rest = tour[:i] + tour[i + seg_len:]
    if len(rest) < 2:
        return None
    p = _rand_int(rng, 0, len(rest))
    new = rest[:p] + segment + rest[p:]
    return new, tour_length(new, coords)


def local_search(tour: list[int], coords: list[list[int]], hp: dict,
                 rng: random.Random) -> list[int]:
    max_iters = max(0, int(hp["max_iterations"]))
    T = float(hp["initial_temperature"])
    cooling = float(hp["cooling_rate"])
    segment_max = max(1, int(hp["segment_max"]))

    cur = tour[:]
    cur_len = tour_length(cur, coords)
    best, best_len = cur[:], cur_len

    def _accepts(delta: int) -> bool:
        if delta < 0:
            return True
        return T > 1e-12 and rng.random() < math.exp(-delta / T)

    for _ in range(max_iters):
        if NEIGHBORHOOD == "or_opt":
            proposal = _or_opt_move(cur, coords, rng, segment_max)
            if proposal is not None:
                new, new_len = proposal
                if _accepts(new_len - cur_len):
                    cur, cur_len = new, new_len
                    if cur_len < best_len:
                        best, best_len = cur[:], cur_len
        else:
            i, j, delta = _two_opt_move(cur, coords, rng)
            if _accepts(delta):
                cur[i:j + 1] = list(reversed(cur[i:j + 1]))
                cur_len += delta
                if cur_len < best_len:
                    best, best_len = cur[:], cur_len
        if T > 1e-12:
            T *= cooling
    return best


def _perturb(tour: list[int], rng: random.Random, strength: int) -> list[int]:
    """Apply `strength` random 2-opt reversals (accepted regardless) to escape
    a local optimum before the next restart (iterated local search)."""
    cur = tour[:]
    n = len(cur)
    for _ in range(max(0, strength)):
        i = _rand_int(rng, 1, n - 2)
        j = _rand_int(rng, i + 1, n - 1)
        cur[i:j + 1] = reversed(cur[i:j + 1])
    return cur


def solve_instance(coords: list[list[int]], hp: dict,
                   rng: random.Random) -> list[int]:
    base = nearest_neighbor(coords) if hp["use_nn_construction"] \
        else list(range(len(coords)))
    best = local_search(base, coords, hp, rng)
    best_len = tour_length(best, coords)
    for _ in range(max(0, int(hp["restarts"]) - 1)):
        start = _perturb(best, rng, int(hp["perturbation_strength"]))
        cand = local_search(start, coords, hp, rng)
        cand_len = tour_length(cand, coords)
        if cand_len < best_len:
            best, best_len = cand, cand_len
    return best


def _load_instances() -> list[dict]:
    path = os.environ.get("AUTORESEARCH_INSTANCES")
    if path:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return load_train()  # public dev surface (manual runs / smoke)


def solve() -> dict:
    started = time.perf_counter()
    hp = dict(HYPERPARAMS)
    if os.environ.get("AUTORESEARCH_SMOKE") == "1":
        hp["max_iterations"] = min(int(hp["max_iterations"]), 200)
        hp["restarts"] = min(int(hp["restarts"]), 1)

    instances = _load_instances()
    rng = random.Random(SOLVER_SEED)
    solutions: dict[str, list[int]] = {}
    reported: dict[str, int] = {}
    for inst in instances:
        coords = inst["coords"]
        if len(coords) != N_CITIES:
            # Solve whatever we are given; the evaluator enforces N_CITIES.
            pass
        tour = solve_instance(coords, hp, rng)
        solutions[inst["instance_id"]] = tour
        reported[inst["instance_id"]] = tour_length(tour, coords)

    return {
        "schema_version": 3,
        "solver": hp,
        "neighborhood": NEIGHBORHOOD,
        "solutions": solutions,
        # Advisory only — the trusted evaluator RECOMPUTES tour lengths and
        # ignores these, so a forged value cannot inflate a score.
        "reported_objectives": reported,
        "solve_seconds": time.perf_counter() - started,
    }


def main() -> None:
    artifact = solve()
    out_dir = ROOT / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "solution.json").write_text(
        json.dumps(artifact, sort_keys=True), encoding="utf-8"
    )
    n = len(artifact["solutions"])
    print(f"solved {n} instance(s) solve_seconds={artifact['solve_seconds']:.3f}")


if __name__ == "__main__":
    main()
