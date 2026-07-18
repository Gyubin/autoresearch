"""Phase 2 drills: blindness, coder confinement, gate exactness, recovery.

Run from the repo root:  uv run python tests/test_phase2.py
These are self-contained checks (no pytest dependency) that exercise the
orchestrator's units directly; the campaign-level E2E lives in the shell
harness. tests/ is outside editable_globs, so a coder worktree write here is
itself a contract violation.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import orchestrator as orch  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "ok  " if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


# ---------------------------------------------------------------------------
# 1. Coder worktree guard hook (pure unit tests — the containment boundary)
# ---------------------------------------------------------------------------

def _decide(guard, tool: str, field: str, value: str) -> str:
    out = asyncio.run(guard({"tool_name": tool, "tool_input": {field: value}},
                            "tid", None))
    return out["hookSpecificOutput"]["permissionDecision"]


def test_coder_guard(tmp: Path) -> None:
    wt = tmp / "wt"
    (wt / "src").mkdir(parents=True)
    (wt / "src" / "train.py").write_text("x = 1\n")
    (tmp / "evaluation").mkdir()
    (tmp / "evaluation" / "heldout_config.json").write_text("{}")
    # a symlink inside src pointing outside
    (wt / "src" / "evil").symlink_to(tmp / "evaluation")

    denials: list[dict] = []
    guard = orch._make_worktree_guard(wt, denials)

    check("guard: Edit inside src allowed",
          _decide(guard, "Edit", "file_path", str(wt / "src" / "train.py")) == "allow")
    check("guard: Write to root src denied (absolute escape)",
          _decide(guard, "Write", "file_path", str(ROOT / "src" / "train.py")) == "deny")
    check("guard: Read of held-out config denied",
          _decide(guard, "Read", "file_path",
                  str(tmp / "evaluation" / "heldout_config.json")) == "deny")
    check("guard: ../ traversal denied",
          _decide(guard, "Edit", "file_path",
                  str(wt / "src" / ".." / ".." / "evaluation" / "x")) == "deny")
    check("guard: symlink-escape via src/evil denied",
          _decide(guard, "Write", "file_path",
                  str(wt / "src" / "evil" / "dataset.py")) == "deny")
    check("guard: Bash denied (not a permitted tool)",
          _decide(guard, "Bash", "command", "cat /etc/passwd") == "deny")
    check("guard: prefix-confusion sibling denied",
          _decide(guard, "Read", "file_path", str(tmp / "wt-evil" / "x")) == "deny")
    check("guard: Read inside worktree allowed",
          _decide(guard, "Read", "file_path", str(wt / "src" / "train.py")) == "allow")
    check("guard: recorded denials", len(denials) >= 5)


# ---------------------------------------------------------------------------
# 2. Generation-grouped stagnation replay (the sharp latent bug)
# ---------------------------------------------------------------------------

def test_stagnation_grouping() -> None:
    def exp(run_id, gen, param, to, decision, verdict="valid_positive"):
        return {"record_type": "experiment", "run_id": run_id, "generation": gen,
                "verdict": verdict, "decision": decision,
                "hypothesis": {"intervention": {"param": param, "from": 0, "to": to}}}

    # One winning generation of 4 (accept + 3 reject) must replay to
    # stagnation 0, not K-1.
    records = [
        exp("r0001", 1, "lr", 1, "accept"),
        exp("r0002", 1, "epochs", 2, "reject"),
        exp("r0003", 1, "l2", 3, "reject"),
        exp("r0004", 1, "momentum", 4, "reject"),
        exp("r0005", 2, "lr", 5, "reject", "valid_inconclusive"),
        exp("r0006", 2, "batch_size", 6, "reject", "valid_inconclusive"),
    ]
    state: dict = {}
    orch.replay_ledger_fields(state, records)
    check("stagnation: winning gen resets, next gen +1", state["stagnation"] == 1,
          f"got {state['stagnation']}")
    check("stagnation: last_accepted is the winner",
          state["last_accepted"]["param"] == "lr")

    # A correction on the winner makes its generation winnerless.
    corrected = records + [{"record_type": "correction", "corrects": "r0001"}]
    state2: dict = {}
    orch.replay_ledger_fields(state2, corrected)
    check("stagnation: corrected winner -> both gens stagnant",
          state2["stagnation"] == 2, f"got {state2['stagnation']}")

    # Legacy Phase-1 records (no generation field) each form a singleton group.
    legacy = [
        {"record_type": "experiment", "run_id": "r1", "verdict": "valid_positive",
         "decision": "accept",
         "hypothesis": {"intervention": {"param": "lr", "from": 0, "to": 1}}},
        {"record_type": "experiment", "run_id": "r2", "verdict": "valid_negative",
         "decision": "reject",
         "hypothesis": {"intervention": {"param": "lr", "from": 1, "to": 2}}},
    ]
    state3: dict = {}
    orch.replay_ledger_fields(state3, legacy)
    check("stagnation: legacy singletons", state3["stagnation"] == 1)


# ---------------------------------------------------------------------------
# 3. distill_insight blindness — never reads gate data, handles coder records
# ---------------------------------------------------------------------------

def test_distill_blindness() -> None:
    gate = {"record_type": "gate", "generation": 1, "incumbent_gate": 0.48,
            "results": {"r0001": 0.37}, "winner": "r0001"}
    check("distill: gate records yield no insight",
          orch.distill_insight(gate) is None)

    # A dev-improver that lost the gate must read as "not admitted", no number.
    rejected = {"record_type": "experiment", "run_id": "r0004", "generation": 1,
                "verdict": "valid_positive", "decision": "reject",
                "best_primary_before": 0.50, "primary": 0.39,
                "hypothesis": {"intervention": {"param": "momentum",
                                                "from": 0.0, "to": 0.9}}}
    ins = orch.distill_insight(rejected)
    check("distill: gate-rejected dev improver flagged",
          ins is not None and "not admitted" in ins["observation"].lower())
    check("distill: no gate score (0.48/0.37) in the insight",
          "0.48" not in json.dumps(ins) and "0.37" not in json.dumps(ins))

    coder = {"record_type": "experiment", "run_id": "r0003", "generation": 1,
             "executor": "coder", "verdict": "valid_positive", "decision": "accept",
             "best_primary_before": 0.50, "primary": 0.25,
             "hypothesis": {"statement": "add x0*x1 interaction feature",
                            "intervention": {"param": None, "from": None,
                                             "to": None, "kind": "coder"}}}
    ins2 = orch.distill_insight(coder)
    check("distill: coder record handled without a param",
          ins2 is not None and ins2["conditions"]["executor"] == "coder")


# ---------------------------------------------------------------------------
# 4. Feature-spec evaluator scoring (the coder's target surface)
# ---------------------------------------------------------------------------

def test_solution_validation() -> None:
    """Phase 6c: the evaluator validates candidate solutions as pure data (a
    permutation per instance). Full feasibility coverage lives in
    tests/test_phase6c.py; this is the Phase-2-level smoke that the evaluator
    imports cleanly and its data-validation seam is wired."""
    ev = orch._load_evaluator_declarations()  # smoke: module imports cleanly
    check("evaluator declares dev/gate/test", ev["split_names"] == ("dev", "gate", "test"))
    check("evaluator declares mean_tour_length/minimize",
          ev["metric_name"] == "mean_tour_length"
          and ev["metric_direction"] == "minimize")

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ev_mod", ROOT / "evaluation" / "evaluate.py")
    ev_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev_mod)

    insts = [{"instance_id": "i0", "coords": [[0, 0], [1, 1], [2, 2], [3, 3]]}]
    notes: list[str] = []
    perms, fail = ev_mod._validate_solutions(insts, {"i0": [0, 1, 2, 3]}, 4, notes)
    check("solution: valid permutation accepted", fail is None and perms == [[0, 1, 2, 3]])
    _, fail2 = ev_mod._validate_solutions(insts, {"i0": [0, 0, 1, 2]}, 4, notes)
    check("solution: non-permutation rejected (infeasible)",
          fail2 == "infeasible_solution")
    _, fail3 = ev_mod._validate_solutions(insts, "not-a-dict", 4, notes)
    check("solution: non-dict solutions rejected (malformed)",
          fail3 == "malformed_solution")


def main() -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        test_coder_guard(Path(td))
    test_stagnation_grouping()
    test_distill_blindness()
    test_solution_validation()

    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s): {FAILS}")
        return 1
    print("all Phase 2 unit drills passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
