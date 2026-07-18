#!/usr/bin/env python3
"""Protected evaluator for the Phase 6c Euclidean-TSP research domain.

PROTECTED FILE (listed in research_contract.yaml protected_globs).

Trust model / invariants:
  * Only the ROOT copy of this file is authoritative. The orchestrator always
    invokes <root>/evaluation/evaluate.py and points it at a candidate workspace
    via --workspace; copies of evaluation/ inside worktrees are never executed
    for scoring.
  * This evaluator trusts nothing outside evaluation/** (plus a hash-only echo of
    the contract): budgets/metric identity are hardcoded here and cross-checked
    against research_contract.yaml at `orchestrator.py init`; the held-out seeds
    (dev/gate/test) come from evaluation/heldout_config.json next to THIS file;
    dataset code is loaded from THIS directory by absolute path.

  * Phase 6c compute-boundary (the domain shift from regression):
      - The task is combinatorial: the candidate is a SOLVER that must run on the
        held-out instances. The evaluator generates the split's instances from
        the hidden seed (TRUSTED), hands the solver ONLY the coordinates (never
        the seed; instance ids are opaque) via the sandbox, and the solver emits
        a tour per instance.
      - The evaluator then VALIDATES each tour is a permutation and RECOMPUTES
        the tour length ITSELF. The solver's self-reported objective is ignored,
        so a forged score cannot inflate a result. Seeds stay in this trusted
        process, still masked out of the container, so a solver cannot regenerate
        other splits.
      - Because the candidate now executes on held-out instances, gate/test
        admission is only fully trustworthy under the CONTAINER backend (the
        subprocess backend has no FS isolation and can read the seed file by
        absolute path). subprocess stays the Docker-free default for dev/smoke
        and tests; the orchestrator warns when it is used for gate/report, and
        sandbox.require_container_for_trusted_splits turns that into a hard error.

  * Phase 6a execution sandbox unchanged: the solver runs under sandbox/runner.py
    (loaded from THIS repo by absolute path); the container backend runs it with
    no network, a read-only rootfs, dropped capabilities, resource limits, an
    ephemeral PID namespace, and the held-out seed file / gate ledger masked.

  * Exit code convention: exit 0 iff a metrics file was written. Scientific
    failures (timeout, crash, infeasible solution, no-skill) still exit 0 with
    executed/failure_class set. A nonzero exit means evaluator infrastructure
    itself broke and the round must be treated as unscored.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

EVALUATOR_VERSION = "2.0.0"
METRICS_SCHEMA_VERSION = "2.0"

# Hardcoded budgets and metric identity — see trust model above. The orchestrator
# cross-checks ALL of these against the contract at init and on every evaluation.
TRAIN_TIMEOUT_S = {"smoke": 30, "dev": 120}
PRIMARY_METRIC_NAME = "mean_tour_length"
PRIMARY_METRIC_DIRECTION = "minimize"
SPLIT_NAMES = ("dev", "gate", "test")

# Phase 5: upper bound on hidden test seeds for multi-seed finalist reproduction.
# Must equal MAX_FINALIST_SEEDS in orchestrator.py; `init` cross-checks the two.
MAX_TEST_SEEDS = 16

# Phase 6a: execution-sandbox backends this evaluator will honor. Must equal
# SUPPORTED_BACKENDS in sandbox/runner.py; `orchestrator.py init` cross-checks.
SUPPORTED_SANDBOX_BACKENDS = ("subprocess", "container")

# Phase 6c: cities per instance. Cross-checked against dataset.N_CITIES at init.
N_CITIES = 60

MAX_ARTIFACT_BYTES = 1_000_000
TAIL_BYTES = 2048

EVAL_DIR = Path(__file__).resolve().parent
ROOT_DIR = EVAL_DIR.parent
HELDOUT_CONFIG = EVAL_DIR / "heldout_config.json"


def _sha256_file(path: Path) -> str | None:
    try:
        with open(path, "rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()
    except OSError:
        return None


def _load_dataset_module():
    """Load the ROOT dataset module by absolute path (never via sys.path)."""
    spec = importlib.util.spec_from_file_location(
        "autoresearch_root_dataset", EVAL_DIR / "dataset.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to build import spec for evaluation/dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SANDBOX_MODULE = None


def _load_sandbox_module():
    """Load the ROOT sandbox runner by absolute path (never via sys.path), so a
    workspace copy can never shadow the isolation code."""
    global _SANDBOX_MODULE
    if _SANDBOX_MODULE is None:
        spec = importlib.util.spec_from_file_location(
            "autoresearch_root_sandbox", EVAL_DIR.parent / "sandbox" / "runner.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("failed to build import spec for sandbox/runner.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _SANDBOX_MODULE = module
    return _SANDBOX_MODULE


def _tail(path: Path) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    return data[-TAIL_BYTES:].decode("utf-8", errors="replace")


def _sanitize(obj: object) -> object:
    """Replace non-finite floats with None so the output is strict JSON."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def _workspace_commit(workspace: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def _atomic_write(out_path: Path, payload: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out_path)


# ---------------------------------------------------------------------------
# Trusted TSP scoring primitives (never import candidate code)
# ---------------------------------------------------------------------------

def _tour_length(perm: list[int], coords: list[list[int]], euclid) -> int:
    n = len(perm)
    return sum(euclid(coords[perm[i]], coords[perm[(i + 1) % n]])
               for i in range(n))


def _nearest_neighbor(coords: list[list[int]], euclid) -> list[int]:
    """Trusted reference construction for the skill floor / gap metric."""
    n = len(coords)
    unvisited = set(range(1, n))
    tour = [0]
    while unvisited:
        last = tour[-1]
        nxt = min(unvisited, key=lambda c: euclid(coords[last], coords[c]))
        tour.append(nxt)
        unvisited.discard(nxt)
    return tour


def _run_train(workspace: Path, mode: str, log_dir: Path, sandbox_cfg,
               run_id: str, split: str, seed_index: int | None,
               instances_path: Path):
    """Run the candidate solver via the configured sandbox on the split's
    instances (coordinates only, no seed). Returns the sandbox's TrainResult."""
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONHASHSEED": "0",
    }
    if mode == "smoke":
        env["AUTORESEARCH_SMOKE"] = "1"

    sandbox_mod = _load_sandbox_module()
    sandbox = sandbox_mod.build_sandbox(sandbox_cfg)
    seed_tag = f"-s{seed_index}" if seed_index is not None else ""
    name_hint = f"{run_id}-{split}{seed_tag}"
    return sandbox.run_train(workspace, mode, env, TRAIN_TIMEOUT_S[mode],
                             log_dir, name_hint, instances_path=instances_path)


def _load_artifact(artifacts_dir: Path, notes: list[str]) -> tuple[dict | None, str | None]:
    """Validate + load solution.json from the sandbox's artifacts dir. Returns
    (artifact, failure_class). Structural problems are malformed_solution."""
    artifact_path = artifacts_dir / "solution.json"
    if artifact_path.is_symlink():
        notes.append("artifact is a symlink; rejected")
        return None, "malformed_solution"
    if not artifact_path.is_file():
        return None, "missing_artifact"
    if artifact_path.stat().st_size > MAX_ARTIFACT_BYTES:
        notes.append(f"artifact exceeds {MAX_ARTIFACT_BYTES} bytes; rejected")
        return None, "malformed_solution"
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        notes.append(f"artifact unreadable: {exc}")
        return None, "malformed_solution"
    if not isinstance(artifact, dict):
        notes.append("artifact is not a JSON object")
        return None, "malformed_solution"
    return artifact, None


def _validate_solutions(
    instances: list[dict], solutions: object, n_cities: int, notes: list[str]
) -> tuple[list[list[int]] | None, str | None]:
    """Validate one tour per instance as pure data. Returns (perms, failure).

    A missing/ill-typed `solutions` object is malformed_solution (structural); a
    well-formed entry that is not a permutation of range(n_cities) is
    infeasible_solution (a scientific verdict — the solver produced an answer,
    just an invalid one)."""
    if not isinstance(solutions, dict):
        notes.append("solutions is not an object")
        return None, "malformed_solution"
    target = set(range(n_cities))
    perms: list[list[int]] = []
    for inst in instances:
        perm = solutions.get(inst["instance_id"])
        if (not isinstance(perm, list) or len(perm) != n_cities
                or not all(isinstance(x, int) and not isinstance(x, bool)
                           for x in perm)
                or set(perm) != target):
            notes.append(
                f"instance {inst['instance_id']}: not a permutation of "
                f"range({n_cities})")
            return None, "infeasible_solution"
        perms.append(perm)
    return perms, None


def evaluate(workspace: Path, mode: str, split: str, run_id: str, nonce: str,
             out_path: Path, sandbox_cfg, seed_index: int | None = None) -> dict:
    started = time.perf_counter()
    out_path.parent.mkdir(parents=True, exist_ok=True)  # log files land here too
    notes: list[str] = []

    metrics: dict = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "mode": mode,
        "split": split,
        "seed_index": seed_index,
        "run_id": run_id,
        "nonce": nonce,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace),
        "workspace_commit": _workspace_commit(workspace),
        "sandbox": {
            "backend": sandbox_cfg.backend,
            "image": sandbox_cfg.image,
            "isolated": sandbox_cfg.backend == "container",
        },
        "executed": False,
        "degenerate": False,
        "failure_class": None,
        "notes": notes,
        "evaluator": {
            "version": EVALUATOR_VERSION,
            "self_sha256": _sha256_file(Path(__file__).resolve()),
            "dataset_sha256": _sha256_file(EVAL_DIR / "dataset.py"),
            "contract_sha256": _sha256_file(ROOT_DIR / "research_contract.yaml"),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "dataset": None,
        "budget": {
            "train_timeout_s": TRAIN_TIMEOUT_S[mode],
            "train_seconds": None,
            "eval_seconds": None,
        },
        "primary_metric": {
            "name": PRIMARY_METRIC_NAME,
            "direction": PRIMARY_METRIC_DIRECTION,
            "value": None,
        },
        "metrics": {},
        "solver": None,
        "train_exit_code": None,
        "stdout_tail": "",
        "stderr_tail": "",
    }

    def done() -> dict:
        metrics["budget"]["eval_seconds"] = round(time.perf_counter() - started, 3)
        return metrics

    if not (workspace / "src" / "train.py").is_file():
        metrics["failure_class"] = "invalid_workspace"
        notes.append("src/train.py not found in workspace")
        return done()
    if not HELDOUT_CONFIG.is_file():
        metrics["failure_class"] = "evaluator_error"
        notes.append("heldout_config.json missing — run `orchestrator.py init` first")
        return done()

    ds = _load_dataset_module()
    if ds.N_CITIES != N_CITIES:
        metrics["failure_class"] = "evaluator_error"
        notes.append(f"N_CITIES drift: evaluator {N_CITIES} vs dataset {ds.N_CITIES}")
        return done()

    effective_index = seed_index or 0
    if split == "test" and effective_index >= MAX_TEST_SEEDS:
        metrics["failure_class"] = "evaluator_error"
        notes.append(f"seed_index {effective_index} exceeds MAX_TEST_SEEDS "
                     f"{MAX_TEST_SEEDS}")
        return done()

    # Generate the split's instances (TRUSTED, from the hidden seed) and hand the
    # solver ONLY the coordinates + opaque ids — never the seed.
    instances = ds.load_split(HELDOUT_CONFIG, split, effective_index)
    handoff = [{"instance_id": i["instance_id"], "coords": i["coords"]}
               for i in instances]
    instances_path = out_path.parent / f"instances_{mode}_{split}_s{effective_index}.json"
    _atomic_write(instances_path, json.dumps(handoff, sort_keys=True))

    metrics["dataset"] = {
        "split": split,
        "seed_index": effective_index,
        "n_instances": len(instances),
        "n_cities": N_CITIES,
        # Proves which instances scored a run (and catches libm drift) without
        # leaking the seed.
        "instances_fingerprint": ds.fingerprint(ds.flat_coords(instances)),
    }

    train_started = time.perf_counter()
    train = _run_train(workspace, mode, out_path.parent, sandbox_cfg,
                       run_id, split, seed_index, instances_path)
    metrics["budget"]["train_seconds"] = round(time.perf_counter() - train_started, 3)
    metrics["train_exit_code"] = train.exit_code
    metrics["stdout_tail"] = _tail(train.stdout_path)
    metrics["stderr_tail"] = _tail(train.stderr_path)

    if train.infra_error:
        raise RuntimeError(
            f"sandbox container failed to launch (docker exit "
            f"{train.exit_code}); see stderr tail: {metrics['stderr_tail'][-500:]}")

    exit_code, timed_out = train.exit_code, train.timed_out
    if timed_out:
        metrics["failure_class"] = "timeout"
        notes.append(f"solve exceeded {TRAIN_TIMEOUT_S[mode]}s; sandbox torn down")
        return done()
    if exit_code != 0:
        metrics["failure_class"] = "nonzero_exit"
        return done()

    metrics["executed"] = True

    artifact, failure = _load_artifact(train.artifacts_dir, notes)
    if failure or artifact is None:
        metrics["failure_class"] = failure or "malformed_solution"
        return done()
    metrics["solver"] = artifact.get("solver")

    perms, failure = _validate_solutions(instances, artifact.get("solutions"),
                                         N_CITIES, notes)
    if failure or perms is None:
        metrics["failure_class"] = failure or "infeasible_solution"
        return done()

    # Trusted objective recomputation — the solver's reported_objectives are
    # IGNORED entirely. euclid comes from the ROOT dataset module.
    euclid = ds.euclid_nint
    per_instance: list[int] = []
    identity_lengths: list[int] = []
    nn_lengths: list[int] = []
    for inst, perm in zip(instances, perms):
        coords = inst["coords"]
        per_instance.append(_tour_length(perm, coords, euclid))
        identity_lengths.append(_tour_length(list(range(N_CITIES)), coords, euclid))
        nn_lengths.append(_tour_length(_nearest_neighbor(coords, euclid),
                                       coords, euclid))

    n = len(per_instance)
    mean_tour_length = sum(per_instance) / n
    mean_identity = sum(identity_lengths) / n
    mean_nn = sum(nn_lengths) / n

    # No-skill floor (constant-baseline analog): a solver no better than visiting
    # cities in index order has no measurable skill.
    if mean_tour_length >= mean_identity:
        metrics["failure_class"] = "no_skill"
        metrics["degenerate"] = True
        notes.append(
            f"mean_tour_length {mean_tour_length:.1f} not better than the "
            f"identity-order tour {mean_identity:.1f}")
        return done()

    metrics["metrics"] = {
        "mean_tour_length": mean_tour_length,
        "mean_identity_length": mean_identity,
        "mean_nn_length": mean_nn,
        # Normalized skill vs the trusted nearest-neighbor reference (display
        # only; admission uses the scalar primary). <0 means beats NN.
        "mean_gap_to_nn": ((mean_tour_length - mean_nn) / mean_nn
                           if mean_nn > 0 else None),
        "feasible_instances": n,
        "solve_seconds": artifact.get("solve_seconds"),
    }
    # Phase 5: per-instance vector for the report's paired bootstrap (test only —
    # dev/gate keep the scalar-only surface, no new blindness surface).
    if split == "test":
        metrics["metrics"]["per_instance_tour_length"] = per_instance

    metrics["primary_metric"]["value"] = mean_tour_length
    return done()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--mode", choices=("smoke", "dev"), default="dev")
    parser.add_argument("--split", choices=SPLIT_NAMES, default="dev")
    parser.add_argument("--seed-index", type=int, default=None)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--nonce", default="")
    parser.add_argument("--sandbox-backend", default="subprocess")
    parser.add_argument("--sandbox-image", default=None)
    parser.add_argument("--sandbox-memory-mb", type=int, default=512)
    parser.add_argument("--sandbox-cpus", type=float, default=1.0)
    parser.add_argument("--sandbox-pids", type=int, default=128)
    args = parser.parse_args()

    if args.mode == "smoke" and args.split != "dev":
        parser.error("--mode smoke is only valid with --split dev")
    if args.seed_index is not None and args.split != "test":
        parser.error("--seed-index is only valid with --split test")
    if args.seed_index is not None and args.seed_index < 0:
        parser.error("--seed-index must be >= 0")
    if args.sandbox_backend not in SUPPORTED_SANDBOX_BACKENDS:
        parser.error(f"--sandbox-backend must be one of "
                     f"{list(SUPPORTED_SANDBOX_BACKENDS)}")

    sandbox_cfg = _load_sandbox_module().SandboxConfig(
        backend=args.sandbox_backend, image=args.sandbox_image,
        memory_mb=args.sandbox_memory_mb, cpus=args.sandbox_cpus,
        pids_limit=args.sandbox_pids,
    )

    try:
        metrics = evaluate(
            args.workspace.resolve(), args.mode, args.split, args.run_id,
            args.nonce, args.out, sandbox_cfg, seed_index=args.seed_index
        )
    except Exception as exc:  # infra failure: report loudly, exit nonzero
        crash = {
            "schema_version": METRICS_SCHEMA_VERSION,
            "mode": args.mode,
            "split": args.split,
            "seed_index": args.seed_index,
            "run_id": args.run_id,
            "nonce": args.nonce,
            "executed": False,
            "degenerate": False,
            "failure_class": "evaluator_error",
            "notes": [f"evaluator crashed: {type(exc).__name__}: {exc}"],
            "primary_metric": {
                "name": PRIMARY_METRIC_NAME,
                "direction": PRIMARY_METRIC_DIRECTION,
                "value": None,
            },
        }
        try:
            _atomic_write(args.out, json.dumps(_sanitize(crash), sort_keys=True))
        except OSError:
            pass
        print(json.dumps(_sanitize(crash), sort_keys=True))
        return 2

    payload = json.dumps(_sanitize(metrics), sort_keys=True, allow_nan=False)
    _atomic_write(args.out, payload)
    print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
