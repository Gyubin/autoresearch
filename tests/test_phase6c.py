"""Phase 6c drills: Euclidean-TSP research domain + the new trust invariants.

Run from the repo root:  uv run python tests/test_phase6c.py
Self-contained, offline, Docker-free (subprocess backend). No SDK call is made.

The domain shift is that the candidate is now a SOLVER that EXECUTES on the
held-out instances, so this suite pins the invariants that keep every prior
guarantee intact:
  * feasibility is validated as pure data (a permutation), and the objective is
    RECOMPUTED in the trusted evaluator — a forged self-report cannot inflate it;
  * the SEED never crosses into the sandbox (the instances handed to the solver
    carry coordinates + opaque ids only);
  * the no-skill floor rejects a degenerate solver;
  * gate-split metrics expose no per-instance vector (no new blindness surface).
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import evaluation.evaluate as ev  # noqa: E402
import orchestrator as orch  # noqa: E402
from evaluation import dataset as ds  # noqa: E402
from sandbox.runner import SandboxConfig  # noqa: E402

FAILS: list[str] = []
DEV_SEED = 700000001   # distinctive: > GRID, cannot be a coordinate value
GATE_SEED = 700000002
TEST_SEEDS = [700000101, 700000102, 700000103]


def check(name: str, cond: bool, detail: str = "") -> None:
    mark = "ok  " if cond else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def _write_heldout(tmp: Path) -> Path:
    cfg = {"schema_version": 4, "splits": {
        "dev": {"seed": DEV_SEED}, "gate": {"seed": GATE_SEED},
        "test": {"seeds": TEST_SEEDS}}}
    p = tmp / "heldout_v4.json"
    p.write_text(json.dumps(cfg))
    return p


# A fake solver workspace: src/train.py reads AUTORESEARCH_INSTANCES and writes
# artifacts/solution.json in one of several MODES, so we can exercise the
# evaluator's validation + trusted recompute against adversarial output.
_FAKE_SOLVER = '''\
import json, math, os
from pathlib import Path
MODE = {mode!r}


def euclid(a, b):
    dx = a[0] - b[0]; dy = a[1] - b[1]
    return int(math.sqrt(dx * dx + dy * dy) + 0.5)


def nn(coords):  # identical to the evaluator's trusted _nearest_neighbor
    n = len(coords); unv = set(range(1, n)); tour = [0]
    while unv:
        last = tour[-1]
        c = min(unv, key=lambda k: euclid(coords[last], coords[k]))
        tour.append(c); unv.discard(c)
    return tour


inst = json.loads(Path(os.environ["AUTORESEARCH_INSTANCES"]).read_text())
solutions, reported = {{}}, {{}}
for i in inst:
    n = len(i["coords"]); iid = i["instance_id"]
    if MODE == "identity":
        perm = list(range(n))          # exactly the no-skill baseline
    elif MODE == "infeasible":
        perm = [0] * n                 # not a permutation (repeats)
    else:                              # honest: a genuine nearest-neighbor tour
        perm = nn(i["coords"])
    solutions[iid] = perm
    reported[iid] = 1                  # a LIE: claim tour length 1 for every instance
Path("artifacts").mkdir(exist_ok=True)
Path("artifacts/solution.json").write_text(json.dumps({{
    "schema_version": 3, "solver": {{}}, "solutions": solutions,
    "reported_objectives": reported, "solve_seconds": 0.0}}))
'''


def _fake_workspace(tmp: Path, mode: str) -> Path:
    ws = tmp / f"ws_{mode}"
    (ws / "src").mkdir(parents=True, exist_ok=True)
    (ws / "src" / "train.py").write_text(_FAKE_SOLVER.format(mode=mode))
    return ws


def _run(ws: Path, heldout: Path, tmp: Path, split: str = "dev",
         seed_index=None) -> dict:
    orig = ev.HELDOUT_CONFIG
    ev.HELDOUT_CONFIG = heldout
    try:
        cfg = SandboxConfig(backend="subprocess", image=None, memory_mb=512,
                            cpus=1.0, pids_limit=128)
        out = tmp / f"metrics_{ws.name}_{split}.json"
        # mode is always "dev" (full budget): gate/test are never smoke.
        return ev.evaluate(ws, "dev", split, "t6c", "nonce", out, cfg,
                           seed_index=seed_index)
    finally:
        ev.HELDOUT_CONFIG = orig


# ---------------------------------------------------------------------------
# Trusted scoring primitives
# ---------------------------------------------------------------------------

def test_scoring_primitives() -> None:
    # A unit square: optimal tour length is 4 * side (perimeter) with nint.
    coords = [[0, 0], [0, 10], [10, 10], [10, 0]]
    perim = ev._tour_length([0, 1, 2, 3], coords, ds.euclid_nint)
    check("scoring: square perimeter tour length", perim == 40, str(perim))
    crossed = ev._tour_length([0, 2, 1, 3], coords, ds.euclid_nint)
    check("scoring: crossing tour is longer", crossed > perim,
          f"{crossed} vs {perim}")
    nn = ev._nearest_neighbor(coords, ds.euclid_nint)
    check("scoring: NN returns a permutation",
          sorted(nn) == [0, 1, 2, 3])


def test_feasibility_validation() -> None:
    insts = [{"instance_id": "i0", "coords": [[0, 0], [1, 1], [2, 2]]}]
    notes: list[str] = []
    perms, fail = ev._validate_solutions(insts, {"i0": [0, 1, 2]}, 3, notes)
    check("feasible: valid permutation accepted", fail is None and perms == [[0, 1, 2]])
    _, fail = ev._validate_solutions(insts, {"i0": [0, 0, 1]}, 3, notes)
    check("feasible: repeat -> infeasible_solution", fail == "infeasible_solution")
    _, fail = ev._validate_solutions(insts, {"i0": [0, 1]}, 3, notes)
    check("feasible: wrong length -> infeasible_solution", fail == "infeasible_solution")
    _, fail = ev._validate_solutions(insts, {"i0": [0, 1, 3]}, 3, notes)
    check("feasible: out-of-range -> infeasible_solution", fail == "infeasible_solution")
    _, fail = ev._validate_solutions(insts, "not-a-dict", 3, notes)
    check("feasible: non-dict solutions -> malformed_solution", fail == "malformed_solution")
    _, fail = ev._validate_solutions(insts, {}, 3, notes)
    check("feasible: missing instance -> infeasible_solution", fail == "infeasible_solution")


def test_trusted_recompute_ignores_forged_report(tmp: Path) -> None:
    heldout = _write_heldout(tmp)
    ws = _fake_workspace(tmp, "honest")
    m = _run(ws, heldout, tmp, "dev")
    check("recompute: run executed", m["executed"] and m["failure_class"] is None,
          str(m["failure_class"]))
    primary = m["primary_metric"]["value"]
    # The fake solver reported length 1 for every instance; the trusted primary
    # must be the recomputed mean (a real, large tour length), NOT the lie.
    check("recompute: forged reported_objectives ignored",
          primary is not None and primary > 100, str(primary))
    # Recompute independently from the instances the evaluator used and confirm
    # the evaluator's primary equals OUR trusted recompute (not the report). The
    # honest fake solver returns a nearest-neighbor tour, which we replicate.
    insts = ds.load_split(heldout, "dev", 0)
    ours = sum(ev._tour_length(ev._nearest_neighbor(i["coords"], ds.euclid_nint),
                               i["coords"], ds.euclid_nint)
               for i in insts) / len(insts)
    check("recompute: evaluator primary == our trusted recompute",
          abs(primary - ours) < 1e-6, f"{primary} vs {ours}")


def test_no_skill_floor(tmp: Path) -> None:
    heldout = _write_heldout(tmp)
    ws = _fake_workspace(tmp, "identity")
    m = _run(ws, heldout, tmp, "dev")
    # An identity-order tour is exactly the no-skill baseline -> rejected.
    check("no_skill: identity-order solver rejected as no_skill",
          m.get("failure_class") == "no_skill" and m.get("degenerate") is True,
          str(m.get("failure_class")))


def test_infeasible_solver(tmp: Path) -> None:
    heldout = _write_heldout(tmp)
    ws = _fake_workspace(tmp, "infeasible")
    m = _run(ws, heldout, tmp, "dev")
    check("infeasible: non-permutation -> infeasible_solution (exit 0, scientific)",
          m.get("failure_class") == "infeasible_solution"
          and m["primary_metric"]["value"] is None,
          str(m.get("failure_class")))


def test_seed_never_enters_sandbox(tmp: Path) -> None:
    """CRITICAL: the instances handed to the solver must carry NO seed value and
    OPAQUE ids — the seed stays in the trusted evaluator (still masked in the
    container)."""
    heldout = _write_heldout(tmp)
    ws = _fake_workspace(tmp, "honest")
    m = _run(ws, heldout, tmp, "dev")
    # The evaluator wrote the handoff file next to the metrics out_path.
    out_dir = (tmp)
    inst_files = list(out_dir.glob("instances_*.json"))
    check("seed-absence: instances handoff file written", len(inst_files) >= 1,
          str([p.name for p in inst_files]))
    if inst_files:
        blob = inst_files[0].read_text()
        check("seed-absence: dev seed value absent from instances handoff",
              str(DEV_SEED) not in blob)
        data = json.loads(blob)
        check("seed-absence: instance ids are opaque (i0, i1, ...)",
              all(rec["instance_id"].startswith("i")
                  and rec["instance_id"][1:].isdigit() for rec in data))
        check("seed-absence: handoff carries only instance_id + coords",
              all(set(rec.keys()) == {"instance_id", "coords"} for rec in data))
    # The dataset fingerprint proves which data scored the run WITHOUT the seed.
    check("seed-absence: dataset fingerprint present, seed not in metrics",
          m["dataset"]["instances_fingerprint"] and str(DEV_SEED) not in json.dumps(m))


def test_gate_split_no_per_instance_surface(tmp: Path) -> None:
    """A solver can compute its own gate tour length, so blindness rests on gate
    RUN OUTPUT never reaching a proposer/insight surface. As a structural guard,
    the gate split must not emit the per-instance vector that the test split
    does (keeping the gate metrics surface scalar-only)."""
    heldout = _write_heldout(tmp)
    ws = _fake_workspace(tmp, "honest")
    m_gate = _run(ws, heldout, tmp, "gate")
    m_test = _run(ws, heldout, tmp, "test", seed_index=0)
    check("gate-blindness: gate metrics omit per_instance_tour_length",
          "per_instance_tour_length" not in (m_gate.get("metrics") or {}))
    check("gate-blindness: test metrics include per_instance_tour_length",
          "per_instance_tour_length" in (m_test.get("metrics") or {}))


def test_n_cities_cross_check() -> None:
    ev_decl = orch._load_evaluator_declarations()["n_cities"]
    ds_decl = orch._load_dataset_declarations()["n_cities"]
    check("cross-check: evaluator N_CITIES == dataset N_CITIES == 60",
          ev_decl == ds_decl == 60, f"{ev_decl}/{ds_decl}")


def test_contract_is_tsp() -> None:
    c = orch.load_contract()
    check("contract: v8 TSP objective/metric",
          c.schema_version == 8
          and c.primary_metric.name == "mean_tour_length"
          and c.primary_metric.direction == "minimize"
          and c.literature.corpus_path == "literature/corpus/tsp_corpus.json")


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_scoring_primitives()
        test_feasibility_validation()
        test_trusted_recompute_ignores_forged_report(tmp)
        test_no_skill_floor(tmp)
        test_infeasible_solver(tmp)
        test_seed_never_enters_sandbox(tmp)
        test_gate_split_no_per_instance_surface(tmp)
        test_n_cities_cross_check()
        test_contract_is_tsp()
    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {FAILS}")
        return 1
    print("all Phase 6c unit drills passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
