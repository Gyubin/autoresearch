#!/usr/bin/env python3
"""AutoResearch Phase 1 orchestrator — constrained keep/reject research loop.

PROTECTED FILE (listed in research_contract.yaml protected_globs).

Implements the Karpathy-style keep/reject routine with basic Arbor-style
state management:

    read contract -> propose ONE falsifiable hypothesis certificate
    -> isolated git worktree -> precise patch of src/train.py's HYPERPARAMS
    -> protected-glob checks (before AND after evaluation)
    -> root evaluator (smoke, then dev) -> classify
    -> write-ahead ledger record -> ff-merge if accepted, else reject
    -> distill insight -> update state -> stop conditions

Design invariants (see README for the full protection model):
  * The contract is immutable: this process only ever READS it. Baseline and
    incumbent results live in experiments/state.json.
  * Only the ROOT evaluator copy scores candidates; worktree copies of
    evaluation/ are data, never code that runs.
  * Provenance: every completed experiment (accepted or rejected) is a
    commit on a retained hyp/* branch plus a ledger record. Runtime state is
    gitignored so main's history contains accepted science only.
  * Repair is allowed ONLY for pre-evaluation mechanical patch failures.
    Runtime failures of a validly patched trainer (divergence, timeout,
    crash) are scientific evidence — valid negative results — because the
    baseline was proven runnable at init and the patcher is deterministic.
    Post-evaluation "repair" is deliberately not implemented: repairing a
    negative result until it turns positive is the classic false-repair
    failure mode.

Usage:
    uv run python orchestrator.py init [--force]
    uv run python orchestrator.py run [--rounds N] [--proposer heuristic|claude]
                                      [--model NAME] [--max-budget-usd X]
    uv run python orchestrator.py status
    uv run python orchestrator.py verify-protection
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import fcntl
import importlib.util
import json
import math
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Optional

import yaml

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
CONTRACT_PATH = ROOT / "research_contract.yaml"
EVALUATOR_PATH = ROOT / "evaluation" / "evaluate.py"
HELDOUT_CONFIG_PATH = ROOT / "evaluation" / "heldout_config.json"
MANIFEST_PATH = ROOT / "protection" / "hashes.json"
TRAIN_REL = "src/train.py"

EXPERIMENTS_DIR = ROOT / "experiments"
ROUNDS_DIR = EXPERIMENTS_DIR / "rounds"
BASELINE_DIR = EXPERIMENTS_DIR / "baseline"
STATE_PATH = EXPERIMENTS_DIR / "state.json"
LEDGER_PATH = EXPERIMENTS_DIR / "ledger.jsonl"
INSIGHTS_PATH = ROOT / "insight_memory.json"
WORKTREES_DIR = ROOT / ".worktrees"
# Outside experiments/ so `init --force` (which clears experiments/) can never
# delete a lock file another process holds.
LOCK_PATH = ROOT / ".orchestrator.lock"

STATE_SCHEMA_VERSION = 2

# Top-level directories never scanned for protected files (runtime/venv/tooling).
SCAN_EXCLUDE = {".git", ".venv", ".worktrees", "experiments", "artifacts",
                "__pycache__", ".omc", ".claude"}
# Directory names excluded at ANY depth, and file suffixes never manifested.
SCAN_EXCLUDE_ANYWHERE = {"__pycache__", ".git"}
SCAN_EXCLUDE_SUFFIXES = (".pyc", ".tmp")


class OrchestratorError(RuntimeError):
    """Base class for expected, user-reportable failures."""


class ContractError(OrchestratorError):
    pass


class ProtectionViolation(OrchestratorError):
    pass


class PatchError(OrchestratorError):
    pass


class GitError(OrchestratorError):
    pass


class EvaluatorInfraError(OrchestratorError):
    pass


class ProposerError(OrchestratorError):
    pass


# ---------------------------------------------------------------------------
# Small IO helpers
# ---------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A crash can leave a torn final line without a newline; gluing the next
    # record onto it would corrupt BOTH records. Heal with a leading newline.
    needs_newline = False
    if path.exists() and path.stat().st_size > 0:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            needs_newline = f.read(1) != b"\n"
    line = json.dumps(obj, sort_keys=True) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        if needs_newline:
            f.write("\n")
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


class InstanceLock:
    """Exclusive advisory lock: two orchestrator processes must never share
    experiments/ — a concurrent run's recovery would destroy the other's
    in-flight worktree and record spurious verdicts."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh = None

    def __enter__(self) -> "InstanceLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")
        try:
            fcntl.flock(self._fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._fh.close()
            raise OrchestratorError(
                "another orchestrator process is running (lock: "
                f"{self.path}); concurrent runs are not allowed"
            ) from None
        self._fh.write(str(os.getpid()))
        self._fh.flush()
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._fh is not None:
            fcntl.flock(self._fh, fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None


def read_jsonl(path: Path) -> list[dict]:
    """Tolerant reader: a crash can truncate the final line; skip and warn."""
    records: list[dict] = []
    if not path.exists():
        return records
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"[warn] {path.name}:{i + 1} is not valid JSON; skipping",
                  file=sys.stderr)
    return records


def sha256_file(path: Path) -> str:
    import hashlib

    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def matches_any(rel_path: str, globs: tuple[str, ...]) -> bool:
    p = PurePosixPath(rel_path)
    return any(p.full_match(g) for g in globs)


def value_repr(v: Any) -> str:
    if isinstance(v, bool):
        return "True" if v else "False"
    return repr(v)


# ---------------------------------------------------------------------------
# Research contract (Layer 1) — typed, versioned, read-only
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PrimaryMetric:
    name: str
    direction: str
    min_relative_improvement: float


@dataclass(frozen=True)
class Budgets:
    smoke_train_timeout_s: int
    dev_train_timeout_s: int
    max_rounds: int
    repair_attempts: int


@dataclass(frozen=True)
class StopConditions:
    stagnation_generations: int


@dataclass(frozen=True)
class Portfolio:
    parallel_branches: int
    gate_top_k: int
    gate_min_relative_improvement: float
    max_coder_hypotheses: int
    max_generations: int | None
    coder_max_turns: int
    coder_max_budget_usd: float


@dataclass(frozen=True)
class ResearchContract:
    schema_version: int
    contract_id: str
    objective: str
    primary_metric: PrimaryMetric
    secondary_metrics: tuple[str, ...]
    editable_globs: tuple[str, ...]
    protected_globs: tuple[str, ...]
    budgets: Budgets
    portfolio: Portfolio
    stop_conditions: StopConditions


def _require(mapping: dict, key: str, kind: type | tuple, ctx: str) -> Any:
    if not isinstance(mapping, dict) or key not in mapping:
        raise ContractError(f"contract: missing '{ctx}.{key}'")
    value = mapping[key]
    if not isinstance(value, kind) or isinstance(value, bool) and kind is not bool:
        raise ContractError(
            f"contract: '{ctx}.{key}' must be {kind}, got {type(value).__name__}"
        )
    return value


def _str_tuple(mapping: dict, key: str, ctx: str) -> tuple[str, ...]:
    value = _require(mapping, key, list, ctx)
    if not value or not all(isinstance(v, str) and v for v in value):
        raise ContractError(f"contract: '{ctx}.{key}' must be a non-empty string list")
    return tuple(value)


def load_contract(path: Path = CONTRACT_PATH) -> ResearchContract:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ContractError(f"contract not found: {path}") from None
    except yaml.YAMLError as exc:
        raise ContractError(f"contract is not valid YAML: {exc}") from None
    if not isinstance(raw, dict):
        raise ContractError("contract root must be a mapping")

    schema_version = _require(raw, "schema_version", int, "")
    if schema_version != 2:
        raise ContractError(f"unsupported contract schema_version {schema_version} "
                            f"(this orchestrator expects 2)")

    pm_raw = _require(raw, "primary_metric", dict, "")
    direction = _require(pm_raw, "direction", str, "primary_metric")
    if direction not in ("minimize", "maximize"):
        raise ContractError("primary_metric.direction must be minimize|maximize")
    min_rel = float(_require(pm_raw, "min_relative_improvement", (int, float),
                             "primary_metric"))
    if not 0.0 < min_rel < 1.0:
        raise ContractError("primary_metric.min_relative_improvement must be in (0, 1)")

    budgets_raw = _require(raw, "budgets", dict, "")
    budgets = Budgets(
        smoke_train_timeout_s=_require(budgets_raw, "smoke_train_timeout_s", int, "budgets"),
        dev_train_timeout_s=_require(budgets_raw, "dev_train_timeout_s", int, "budgets"),
        max_rounds=_require(budgets_raw, "max_rounds", int, "budgets"),
        repair_attempts=_require(budgets_raw, "repair_attempts", int, "budgets"),
    )
    if min(budgets.smoke_train_timeout_s, budgets.dev_train_timeout_s,
           budgets.max_rounds) <= 0 or budgets.repair_attempts < 0:
        raise ContractError("contract budgets must be positive")

    stop_raw = _require(raw, "stop_conditions", dict, "")
    stop = StopConditions(
        stagnation_generations=_require(stop_raw, "stagnation_generations", int,
                                        "stop_conditions")
    )
    if stop.stagnation_generations <= 0:
        raise ContractError("stop_conditions.stagnation_generations must be positive")

    pf_raw = _require(raw, "portfolio", dict, "")
    max_generations = pf_raw.get("max_generations")
    if max_generations is not None and (
        not isinstance(max_generations, int) or isinstance(max_generations, bool)
        or max_generations <= 0
    ):
        raise ContractError("portfolio.max_generations must be a positive int or null")
    portfolio = Portfolio(
        parallel_branches=_require(pf_raw, "parallel_branches", int, "portfolio"),
        gate_top_k=_require(pf_raw, "gate_top_k", int, "portfolio"),
        gate_min_relative_improvement=float(
            _require(pf_raw, "gate_min_relative_improvement", (int, float), "portfolio")
        ),
        max_coder_hypotheses=_require(pf_raw, "max_coder_hypotheses", int, "portfolio"),
        max_generations=max_generations,
        coder_max_turns=_require(pf_raw, "coder_max_turns", int, "portfolio"),
        coder_max_budget_usd=float(
            _require(pf_raw, "coder_max_budget_usd", (int, float), "portfolio")
        ),
    )
    if not 1 <= portfolio.gate_top_k <= portfolio.parallel_branches <= 8:
        raise ContractError(
            "portfolio must satisfy 1 <= gate_top_k <= parallel_branches <= 8"
        )
    if not 0 <= portfolio.max_coder_hypotheses <= portfolio.parallel_branches:
        raise ContractError(
            "portfolio.max_coder_hypotheses must be within [0, parallel_branches]"
        )
    if not 0.0 <= portfolio.gate_min_relative_improvement < 1.0:
        raise ContractError(
            "portfolio.gate_min_relative_improvement must be in [0, 1)"
        )

    return ResearchContract(
        schema_version=schema_version,
        contract_id=_require(raw, "contract_id", str, ""),
        objective=_require(raw, "objective", str, "").strip(),
        primary_metric=PrimaryMetric(
            name=_require(pm_raw, "name", str, "primary_metric"),
            direction=direction,
            min_relative_improvement=min_rel,
        ),
        secondary_metrics=_str_tuple(raw, "secondary_metrics", ""),
        editable_globs=_str_tuple(raw, "editable_globs", ""),
        protected_globs=_str_tuple(raw, "protected_globs", ""),
        budgets=budgets,
        portfolio=portfolio,
        stop_conditions=stop,
    )


# ---------------------------------------------------------------------------
# Protection guard: SHA-256 manifest + glob checks
# ---------------------------------------------------------------------------

MANIFEST_REL = "protection/hashes.json"


class ProtectionGuard:
    """SHA-256 manifest over protected files.

    The manifest cannot contain its own hash (self-reference), so its own
    integrity is checked differently: it is git-tracked, listed under a
    protected glob (worktree diffs against it are contract violations), and
    verify() additionally compares the working copy against the committed
    blob via `git diff`.
    """

    def __init__(self, root: Path, contract: ResearchContract) -> None:
        self.root = root
        self.contract = contract

    def _iter_root_files(self):
        for dirpath, dirnames, filenames in os.walk(self.root):
            rel_dir = Path(dirpath).relative_to(self.root)
            at_top = not rel_dir.parts
            dirnames[:] = [
                d for d in dirnames
                if d not in SCAN_EXCLUDE_ANYWHERE
                and not (at_top and d in SCAN_EXCLUDE)
            ]
            for name in filenames:
                if name.endswith(SCAN_EXCLUDE_SUFFIXES):
                    continue
                yield (Path(dirpath) / name).relative_to(self.root).as_posix()

    def protected_files(self) -> list[str]:
        return sorted(
            rel for rel in self._iter_root_files()
            if matches_any(rel, self.contract.protected_globs)
        )

    def build_manifest(self) -> dict[str, str]:
        return {rel: sha256_file(self.root / rel)
                for rel in self.protected_files() if rel != MANIFEST_REL}

    def write_manifest(self) -> None:
        atomic_write_json(MANIFEST_PATH, {
            "algorithm": "sha256",
            "created_utc": utc_now(),
            "protected_globs": list(self.contract.protected_globs),
            "files": self.build_manifest(),
        })

    def load_manifest(self) -> dict:
        try:
            manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProtectionViolation(f"protection manifest unreadable: {exc}") from None
        if not isinstance(manifest.get("files"), dict):
            raise ProtectionViolation("protection manifest malformed")
        return manifest

    def verify(self) -> list[str]:
        manifest = self.load_manifest()
        recorded: dict[str, str] = manifest["files"]
        current = self.build_manifest()
        violations = []
        for rel, digest in recorded.items():
            if rel not in current:
                violations.append(f"protected file missing: {rel}")
            elif current[rel] != digest:
                violations.append(f"protected file modified: {rel}")
        for rel in current:
            if rel not in recorded:
                violations.append(f"unexpected new protected-path file: {rel}")
        proc = subprocess.run(
            ["git", "-C", str(self.root), "diff", "--quiet", "HEAD", "--",
             MANIFEST_REL],
            capture_output=True,
        )
        if proc.returncode == 1:
            violations.append("protection manifest differs from committed version")
        return violations

    def set_read_only(self) -> None:
        for rel in self.protected_files():
            os.chmod(self.root / rel, 0o444)


# ---------------------------------------------------------------------------
# Git operations
# ---------------------------------------------------------------------------

class Git:
    """Git wrapper with a concurrency policy for the parallel portfolio.

    Worker threads may only run per-worktree operations scoped to THEIR OWN
    worktree (add/commit/diff/status/checkout/rev-parse): linked worktrees
    have private index and HEAD files, and commits take per-ref locks on
    distinct hyp/* refs. Operations that mutate SHARED repo state — the
    .git/worktrees admin dir (worktree add/remove/prune), packed-refs
    (branch -D), and main's HEAD (merge) — are serialized behind
    _mutation_lock and, by construction, called from the main thread only.
    """

    _mutation_lock = threading.Lock()

    def __init__(self, root: Path) -> None:
        self.root = root

    def run(self, *args: str, cwd: Path | None = None,
            check: bool = True) -> subprocess.CompletedProcess:
        proc = subprocess.run(
            ["git", "-C", str(cwd or self.root), *args],
            capture_output=True, text=True,
        )
        if check and proc.returncode != 0:
            raise GitError(
                f"git {' '.join(args)} failed ({proc.returncode}): "
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )
        return proc

    def is_repo(self) -> bool:
        return self.run("rev-parse", "--git-dir", check=False).returncode == 0

    def has_head(self) -> bool:
        return self.run("rev-parse", "--verify", "HEAD", check=False).returncode == 0

    def head(self, cwd: Path | None = None) -> str:
        return self.run("rev-parse", "HEAD", cwd=cwd).stdout.strip()

    def init_repo(self) -> None:
        self.run("init", "-b", "main")

    def commit_all(self, message: str, cwd: Path | None = None) -> str:
        self.run("add", "-A", cwd=cwd)
        self.run("commit", "-m", message, cwd=cwd)
        return self.head(cwd=cwd)

    def branch_exists(self, name: str) -> bool:
        return self.run("rev-parse", "--verify", f"refs/heads/{name}",
                        check=False).returncode == 0

    def delete_branch(self, name: str) -> None:
        with self._mutation_lock:
            self.run("branch", "-D", name)

    def branch_tip(self, name: str) -> str:
        return self.run("rev-parse", f"refs/heads/{name}").stdout.strip()

    def worktree_add(self, path: Path, branch: str, base: str) -> None:
        with self._mutation_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.run("worktree", "add", "-b", branch, str(path), base)

    def worktree_add_detached(self, path: Path, commit: str) -> None:
        with self._mutation_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.run("worktree", "add", "--detach", str(path), commit)

    def worktree_remove(self, path: Path) -> None:
        with self._mutation_lock:
            proc = self.run("worktree", "remove", "--force", str(path), check=False)
            if proc.returncode != 0 and path.exists():
                shutil.rmtree(path, ignore_errors=True)
            self.run("worktree", "prune", check=False)

    def worktree_prune(self) -> None:
        with self._mutation_lock:
            self.run("worktree", "prune", check=False)

    def status_paths(self, cwd: Path | None = None,
                     include_untracked: bool = True) -> list[str]:
        args = ["status", "--porcelain=v1"]
        if not include_untracked:
            args.append("--untracked-files=no")
        out = self.run(*args, cwd=cwd).stdout
        paths = []
        for line in out.splitlines():
            if len(line) < 4:
                continue
            entry = line[3:]
            if " -> " in entry:
                # A rename touches BOTH paths; report both so a protected
                # file renamed into an editable location is still flagged.
                old, new = entry.split(" -> ", 1)
                paths.append(old.strip().strip('"'))
                entry = new
            paths.append(entry.strip().strip('"'))
        return paths

    def diff_paths(self, base: str, cwd: Path | None = None) -> list[str]:
        # --no-renames: a rename must surface as delete+add so both the old
        # (possibly protected) and new path go through the glob checks.
        out = self.run("diff", "--name-only", "--no-renames", f"{base}..HEAD",
                       cwd=cwd).stdout
        return [line.strip() for line in out.splitlines() if line.strip()]

    def merge_ff(self, ref: str) -> None:
        with self._mutation_lock:
            self.run("merge", "--ff-only", ref)

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        return self.run("merge-base", "--is-ancestor", ancestor, descendant,
                        check=False).returncode == 0


# ---------------------------------------------------------------------------
# Hypothesis certificate + patcher (the deterministic Phase 1 "coding worker")
# ---------------------------------------------------------------------------

@dataclass
class Hypothesis:
    id: str
    round: int
    statement: str
    mechanism: str
    intervention: dict
    predicted_effect: str
    falsifier: str
    minimal_test: str
    proposer: str
    executor: str = "patcher"  # "patcher" | "coder"
    implementation_brief: str = ""  # coder-only: what to build under src/**
    prior_evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _slug(hypothesis: Hypothesis) -> str:
    base = hypothesis.intervention.get("param") or hypothesis.executor
    return re.sub(r"[^a-z0-9_-]", "-", str(base).lower())[:30] or "hyp"


BEGIN_MARK = "# --- HYPERPARAMS-BEGIN"
END_MARK = "# --- HYPERPARAMS-END"


class HyperparamsPatcher:
    """Applies exactly one hyperparameter change inside the marker block.

    Repair strategy escalation (bounded by contract budgets.repair_attempts):
      attempt 0  targeted single-line regex substitution
      attempt 1+ regenerate the whole dict literal between the markers
    Both paths must survive ast.parse + a literal round-trip equality check.
    This is the seam where a Phase 2 LLM coding worker plugs in.
    """

    @staticmethod
    def _split_block(text: str) -> tuple[list[str], list[str], list[str]]:
        lines = text.splitlines(keepends=True)
        begin = [i for i, l in enumerate(lines) if l.lstrip().startswith(BEGIN_MARK)]
        end = [i for i, l in enumerate(lines) if l.lstrip().startswith(END_MARK)]
        if len(begin) != 1 or len(end) != 1 or begin[0] >= end[0]:
            raise PatchError("HYPERPARAMS marker block missing, duplicated, or inverted")
        b, e = begin[0], end[0]
        return lines[: b + 1], lines[b + 1 : e], lines[e:]

    @classmethod
    def read(cls, train_path: Path) -> dict:
        try:
            text = train_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise PatchError(f"cannot read {train_path}: {exc}") from None
        _, block, _ = cls._split_block(text)
        try:
            module = ast.parse("".join(block))
        except SyntaxError as exc:
            raise PatchError(f"HYPERPARAMS block does not parse: {exc}") from None
        for node in module.body:
            if (isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "HYPERPARAMS"):
                try:
                    params = ast.literal_eval(node.value)
                except ValueError as exc:
                    raise PatchError(f"HYPERPARAMS is not a literal: {exc}") from None
                if not isinstance(params, dict):
                    raise PatchError("HYPERPARAMS is not a dict literal")
                return params
        raise PatchError("no HYPERPARAMS assignment inside marker block")

    @classmethod
    def apply(cls, train_path: Path, param: str, new_value: Any, attempt: int) -> None:
        current = cls.read(train_path)
        if param not in current:
            raise PatchError(f"unknown hyperparameter: {param}")
        expected = dict(current)
        expected[param] = new_value

        text = train_path.read_text(encoding="utf-8")
        head, block, tail = cls._split_block(text)

        if attempt == 0:
            pattern = re.compile(
                rf'^(\s*)(["\']){re.escape(param)}\2\s*:\s*.+?,\s*$', re.MULTILINE
            )
            block_text = "".join(block)
            hits = pattern.findall(block_text)
            if len(hits) != 1:
                raise PatchError(
                    f"expected exactly 1 line for {param!r}, found {len(hits)}"
                )
            new_block = pattern.sub(
                lambda m: f'{m.group(1)}"{param}": {value_repr(new_value)},',
                block_text, count=1,
            )
        else:
            indent = "    "
            body = [f"HYPERPARAMS = {{\n"]
            body += [f"{indent}{k!r}: {value_repr(v)},\n" for k, v in expected.items()]
            body.append("}\n")
            new_block = "".join(body)

        new_text = "".join(head) + new_block + "".join(tail)
        try:
            ast.parse(new_text)
        except SyntaxError as exc:
            raise PatchError(f"patched file does not parse: {exc}") from None

        train_path.write_text(new_text, encoding="utf-8")

        reread = cls.read(train_path)
        if reread != expected:
            raise PatchError(
                f"post-patch mismatch: expected {expected!r}, got {reread!r}"
            )


# ---------------------------------------------------------------------------
# Proposers
# ---------------------------------------------------------------------------

@dataclass
class ProposalContext:
    contract: ResearchContract
    round_index: int
    current_hyperparams: dict
    best_primary: float | None
    tested: dict[str, list[str]]
    last_accepted: dict | None
    insights: list[dict]


_MECHANISMS = {
    "lr_up": "Training appears optimizer-limited within the epoch budget; a "
             "larger step size reaches a lower loss basin in the same number "
             "of updates — until it crosses the stability threshold.",
    "lr_down": "If the current step size overshoots along ill-conditioned "
               "directions, a smaller one should converge more stably.",
    "epochs_up": "The loss curve has not plateaued; more passes convert "
                 "compute directly into fit.",
    "feature_scaling": "Features have heterogeneous scales (condition number "
                       "~625 unscaled); standardization equalizes curvature "
                       "so a single global learning rate suits all "
                       "coordinates.",
    "momentum": "Heavy-ball momentum accelerates progress along "
                "low-curvature directions; the effective step grows to "
                "~lr/(1-beta), so it acts like a cheap lr increase with "
                "smoothing.",
    "l2": "If the train/held-out gap indicates variance, weight decay trades "
          "a little bias for it.",
    "batch_down": "Smaller batches take more update steps per epoch at "
                  "similar gradient quality — more optimization progress per "
                  "epoch.",
    "batch_up": "Larger batches reduce gradient noise; helps iff noise, not "
                "step count, is the binding constraint.",
}


def _round_sig(x: float, digits: int = 6) -> float:
    return float(f"{x:.{digits}g}")


class HeuristicProposer:
    """Deterministic rule-based proposer.

    Move policy: a static priority order over parameter moves, with
    'momentum of search' — the move kind that produced the last accepted
    round is retried first. Tested (param, value) pairs are never
    re-proposed. Returns None when the local move space is exhausted.
    """

    name = "heuristic"

    def _moves(self, hp: dict) -> list[tuple[str, str, Any]]:
        moves: list[tuple[str, str, Any]] = []  # (kind, param, new_value)
        moves.append(("feature_scaling", "feature_scaling", not hp["feature_scaling"]))
        moves.append(("lr_up", "lr", _round_sig(float(hp["lr"]) * 2.5)))
        moves.append(("epochs_up", "epochs", min(int(hp["epochs"]) * 2, 400)))
        for beta in (0.9, 0.5, 0.95):
            if not math.isclose(float(hp["momentum"]), beta):
                moves.append(("momentum", "momentum", beta))
        moves.append(("lr_down", "lr", _round_sig(float(hp["lr"]) / 2.5)))
        moves.append(("batch_down", "batch_size", max(int(hp["batch_size"]) // 2, 4)))
        for lam in (0.001, 0.0001, 0.01):
            if not math.isclose(float(hp["l2"]), lam):
                moves.append(("l2", "l2", lam))
        moves.append(("batch_up", "batch_size", min(int(hp["batch_size"]) * 2, 256)))
        return moves

    def _filtered_candidates(self, ctx: ProposalContext) -> list[tuple[str, str, Any]]:
        hp = ctx.current_hyperparams
        candidates = []
        for kind, param, new_value in self._moves(hp):
            if new_value == hp[param]:
                continue
            if isinstance(new_value, float) and (
                param == "lr" and not 1e-5 <= new_value <= 5.0
            ):
                continue
            if value_repr(new_value) in ctx.tested.get(param, []):
                continue
            candidates.append((kind, param, new_value))

        last = ctx.last_accepted
        if last:
            for cand in candidates:
                if cand[0] == last.get("kind"):
                    candidates.remove(cand)
                    candidates.insert(0, cand)
                    break
        return candidates

    def _certificate(self, ctx: ProposalContext, round_index: int,
                     kind: str, param: str, new_value: Any) -> Hypothesis:
        hp = ctx.current_hyperparams
        pm = ctx.contract.primary_metric
        best = ctx.best_primary
        best_txt = f"{best:.4f}" if best is not None else "the baseline"
        return Hypothesis(
            id=f"h_r{round_index:04d}_{param}",
            round=round_index,
            statement=(
                f"Changing {param} from {hp[param]!r} to {new_value!r} will "
                f"{pm.direction} {pm.name} by at least "
                f"{pm.min_relative_improvement:.1%} relative to the incumbent."
            ),
            mechanism=_MECHANISMS[kind],
            intervention={"param": param, "from": hp[param], "to": new_value,
                          "kind": kind},
            predicted_effect=(
                f"{pm.name} improves from {best_txt} by >= "
                f"{pm.min_relative_improvement:.1%} under the deterministic "
                f"dev evaluation."
            ),
            falsifier=(
                f"{pm.name} fails to improve by "
                f">= {pm.min_relative_improvement:.1%} (or the run becomes "
                f"degenerate) in the deterministic dev evaluation — the "
                f"single decisive test, since evaluation is seeded and "
                f"deterministic."
            ),
            minimal_test="one smoke + one dev evaluation on the patched worktree",
            proposer=self.name,
            executor="patcher",
            prior_evidence=[i["insight_id"] for i in ctx.insights[-3:]
                            if "insight_id" in i],
        )

    def propose_batch(self, ctx: ProposalContext, k: int) -> list[Hypothesis]:
        """Top-k diverse moves: at most one hypothesis per parameter."""
        batch: list[Hypothesis] = []
        seen_params: set[str] = set()
        for kind, param, new_value in self._filtered_candidates(ctx):
            if param in seen_params:
                continue
            batch.append(self._certificate(ctx, ctx.round_index + len(batch),
                                           kind, param, new_value))
            seen_params.add(param)
            if len(batch) == k:
                break
        return batch


class ClaudeProposer:
    """LLM-backed proposer via the Claude Agent SDK (no headless CLI calls;
    the SDK bundles and manages the Claude Code runtime itself).

    Tools are fully disabled and setting_sources=[] so the proposer is a pure
    text-in/JSON-out reasoning call with no filesystem or settings access.
    """

    name = "claude"

    def __init__(self, model: str | None = None, max_budget_usd: float = 0.5) -> None:
        self.model = model
        self.max_budget_usd = max_budget_usd
        self.last_cost_usd: float | None = None

    def _schema(self, hp: dict, k: int, allow_coder: bool) -> dict:
        executors = ["patcher", "coder"] if allow_coder else ["patcher"]
        item = {
            "type": "object",
            "properties": {
                "statement": {"type": "string"},
                "mechanism": {"type": "string"},
                "executor": {"type": "string", "enum": executors},
                # anyOf (not union type lists): the SDK's strict-mode schema
                # validator rejects `"type": [...]` unions.
                "param": {"anyOf": [{"type": "string", "enum": sorted(hp.keys())},
                                    {"type": "null"}]},
                "new_value": {"anyOf": [{"type": "number"}, {"type": "boolean"},
                                        {"type": "null"}]},
                "implementation_brief": {"anyOf": [{"type": "string"},
                                                   {"type": "null"}]},
                "predicted_effect": {"type": "string"},
                "falsifier": {"type": "string"},
            },
            "required": ["statement", "mechanism", "executor", "param",
                         "new_value", "implementation_brief",
                         "predicted_effect", "falsifier"],
            "additionalProperties": False,
        }
        return {
            "type": "object",
            "properties": {
                "hypotheses": {"type": "array", "minItems": 1, "maxItems": k,
                               "items": item},
            },
            "required": ["hypotheses"],
            "additionalProperties": False,
        }

    def _prompt(self, ctx: ProposalContext, k: int, max_coder: int,
                feedback: str | None) -> str:
        pm = ctx.contract.primary_metric
        insights = ctx.insights[-10:]
        tested_lines = [
            f"  {param}: {', '.join(values)}"
            for param, values in sorted(ctx.tested.items())
        ] or ["  (none yet)"]
        parts = [
            "You are the hypothesis proposer inside a constrained autoresearch "
            "loop (portfolio of parallel keep/reject experiments).",
            f"Objective: {ctx.contract.objective}",
            f"Primary metric: {pm.name} ({pm.direction}); a candidate must "
            f"improve the incumbent by >= {pm.min_relative_improvement:.1%} "
            f"relative on the dev split, then survive a blind admission gate. "
            f"Evaluation is deterministic (fixed seeds).",
            f"Current incumbent {pm.name} (dev): {ctx.best_primary!r}",
            f"Current hyperparameters of src/train.py (minibatch SGD over a "
            f"declarative feature spec; 8 raw features with heterogeneous "
            f"scales): {json.dumps(ctx.current_hyperparams)}",
            "Already-tested interventions — do NOT repeat any of these "
            "(param: values):",
            *tested_lines,
            "Distilled insights from past rounds:",
            json.dumps(insights, indent=2) if insights else "  (none yet)",
            f"Propose UP TO {k} scientifically DIVERSE hypotheses for one "
            f"portfolio generation. Rules:\n"
            f"- executor='patcher': change exactly ONE hyperparameter to ONE "
            f"new value (set param + new_value; implementation_brief null).\n"
            f"- executor='coder' (at most {max_coder} per generation): a code "
            f"change under src/** implemented by a coding agent (set param "
            f"and new_value to null; write a concrete implementation_brief). "
            f"The trainer may declare engineered features in "
            f"artifacts/model.json as 'feature_spec': a list of terms, each "
            f"a list of raw-feature indices multiplied together (e.g. "
            f"[[0],[1],...,[7],[0,1]] adds an x0*x1 interaction input). The "
            f"evaluator scores bias + weights . engineered_features. If the "
            f"target relationship has structure a linear readout of raw "
            f"features cannot capture, a coder hypothesis extending the "
            f"feature spec is the only way to reach it.\n"
            f"- No two hypotheses on the same param; prefer mechanisms that "
            f"attack DIFFERENT bottlenecks (optimization, model class, "
            f"regularization).",
        ]
        if feedback:
            parts.append(f"Your previous batch had problems: {feedback}. "
                         f"Propose a corrected batch.")
        return "\n\n".join(parts)

    def _query(self, prompt: str, schema: dict) -> dict:
        import asyncio

        from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

        options = ClaudeAgentOptions(
            tools=[],
            setting_sources=[],
            max_turns=1,
            model=self.model,
            system_prompt=(
                "You are a careful ML research strategist. Respond only with "
                "the requested structured output."
            ),
            output_format={"type": "json_schema", "schema": schema},
            max_budget_usd=self.max_budget_usd,
        )

        async def _run() -> dict:
            result: dict | None = None
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, ResultMessage):
                    self.last_cost_usd = message.total_cost_usd
                    if message.is_error:
                        raise ProposerError(f"SDK returned error: {message.result}")
                    if isinstance(message.structured_output, dict):
                        result = message.structured_output
                    elif message.result:
                        result = json.loads(message.result)
            if result is None:
                raise ProposerError("no structured output from Claude Agent SDK")
            return result

        return asyncio.run(_run())

    def _validate_item(self, raw: dict, ctx: ProposalContext,
                       seen_params: set[str], coder_count: int,
                       max_coder: int) -> Hypothesis | str:
        """Returns a Hypothesis or an error string."""
        hp = ctx.current_hyperparams
        executor = raw.get("executor")

        if executor == "coder":
            if coder_count >= max_coder:
                return f"more than {max_coder} coder hypotheses"
            brief = raw.get("implementation_brief")
            if not isinstance(brief, str) or len(brief.strip()) < 30:
                return "coder hypothesis needs a concrete implementation_brief"
            return Hypothesis(
                id=f"h_r{ctx.round_index:04d}_coder",
                round=ctx.round_index,
                statement=str(raw["statement"]),
                mechanism=str(raw["mechanism"]),
                intervention={"param": None, "from": None, "to": None,
                              "kind": "coder"},
                predicted_effect=str(raw["predicted_effect"]),
                falsifier=str(raw["falsifier"]),
                minimal_test="one smoke + one dev evaluation on the coder's worktree",
                proposer=self.name,
                executor="coder",
                implementation_brief=brief.strip(),
                prior_evidence=[i["insight_id"] for i in ctx.insights[-3:]
                                if "insight_id" in i],
            )

        param = raw.get("param")
        if not isinstance(param, str) or param not in hp:
            return f"unknown param {param!r}"
        if param in seen_params:
            return f"duplicate param {param!r} in one generation"
        new_value: Any = raw.get("new_value")
        current = hp[param]
        if isinstance(current, bool):
            if not isinstance(new_value, bool):
                return f"{param} expects a boolean"
        elif isinstance(current, int) and param in ("epochs", "batch_size"):
            if isinstance(new_value, bool) or not isinstance(new_value, (int, float)):
                return f"{param} expects an integer"
            new_value = int(round(new_value))
            if new_value <= 0:
                return f"{param} must be positive"
        else:
            if isinstance(new_value, bool) or not isinstance(new_value, (int, float)):
                return f"{param} expects a number"
            new_value = float(new_value)
        if new_value == current:
            return f"{param} unchanged ({current!r})"
        if value_repr(new_value) in ctx.tested.get(param, []):
            return f"({param}, {new_value!r}) was already tested"
        return Hypothesis(
            id=f"h_r{ctx.round_index:04d}_{param}",
            round=ctx.round_index,
            statement=str(raw["statement"]),
            mechanism=str(raw["mechanism"]),
            intervention={"param": param, "from": current, "to": new_value,
                          "kind": f"claude:{param}"},
            predicted_effect=str(raw["predicted_effect"]),
            falsifier=str(raw["falsifier"]),
            minimal_test="one smoke + one dev evaluation on the patched worktree",
            proposer=self.name,
            executor="patcher",
            prior_evidence=[i["insight_id"] for i in ctx.insights[-3:]
                            if "insight_id" in i],
        )

    def propose_batch(self, ctx: ProposalContext, k: int) -> list[Hypothesis]:
        max_coder = ctx.contract.portfolio.max_coder_hypotheses
        schema = self._schema(ctx.current_hyperparams, k, max_coder > 0)
        feedback: str | None = None
        for _ in range(2):  # one whole-batch retry with per-item feedback
            raw = self._query(self._prompt(ctx, k, max_coder, feedback), schema)
            items = raw.get("hypotheses")
            if not isinstance(items, list):
                feedback = "missing hypotheses array"
                continue
            batch: list[Hypothesis] = []
            seen_params: set[str] = set()
            coder_count = 0
            errors: list[str] = []
            for item in items[:k]:
                validated = self._validate_item(
                    item if isinstance(item, dict) else {},
                    ctx, seen_params, coder_count, max_coder,
                )
                if isinstance(validated, str):
                    errors.append(validated)
                    continue
                if validated.executor == "coder":
                    coder_count += 1
                else:
                    seen_params.add(validated.intervention["param"])
                batch.append(validated)
            if batch:
                return batch
            feedback = "; ".join(errors) or "no valid items"
        raise ProposerError(f"Claude proposer produced no valid batch: {feedback}")


class FallbackProposer:
    """Primary proposer with heuristic fallback (never blocks the loop)."""

    def __init__(self, primary, fallback: HeuristicProposer) -> None:
        self.primary = primary
        self.fallback = fallback
        self.name = f"{primary.name}+fallback"

    @property
    def last_cost_usd(self) -> float | None:
        return getattr(self.primary, "last_cost_usd", None)

    def propose_batch(self, ctx: ProposalContext, k: int) -> list[Hypothesis]:
        batch: list[Hypothesis] = []
        try:
            batch = self.primary.propose_batch(ctx, k)
        except Exception as exc:  # SDK/network/validation failure
            print(f"[warn] {self.primary.name} proposer failed ({exc}); "
                  f"falling back to heuristic", file=sys.stderr)
        if len(batch) < k:
            used = {h.intervention.get("param") for h in batch
                    if h.intervention.get("param") is not None}
            for hyp in self.fallback.propose_batch(ctx, k):
                if hyp.intervention.get("param") in used:
                    continue
                batch.append(hyp)
                used.add(hyp.intervention.get("param"))
                if len(batch) == k:
                    break
        return batch


# ---------------------------------------------------------------------------
# LLM coding worker (ClaudeCoder executor)
# ---------------------------------------------------------------------------

class CoderError(OrchestratorError):
    """Mechanical failure of the coding-agent call itself."""


_TOOL_PATH_FIELDS = {
    "Read": "file_path",
    "Write": "file_path",
    "Edit": "file_path",
    "Glob": "path",
    "Grep": "path",
}


def _make_worktree_guard(worktree: Path, denials: list[dict]):
    """PreToolUse hook confining the coder agent to its worktree.

    cwd does NOT confine SDK tools (they accept absolute paths), so this hook
    is the sole allower: every tool call gets an explicit allow/deny. Reads
    are confined to the worktree — an unconfined Read would reach the root's
    heldout_config.json (hidden seeds) and gate metrics. Writes/Edits are
    confined to <worktree>/src. Paths are resolved via Path.resolve() (kills
    `..` and symlink tricks) and compared with is_relative_to, never string
    prefixes. Combined with permission_mode="dontAsk" this fails CLOSED: if
    the hook errors or times out, the call falls through to a mode that
    denies by default.
    """
    wt = worktree.resolve()
    src = (wt / "src").resolve()

    async def guard(hook_input, tool_use_id, context):  # noqa: ANN001
        tool = (hook_input or {}).get("tool_name")
        tool_input = (hook_input or {}).get("tool_input") or {}

        def decision(allowed: bool, reason: str) -> dict:
            if not allowed:
                denials.append({"tool": tool, "reason": reason,
                                "input": {k: v for k, v in tool_input.items()
                                          if isinstance(v, str)}})
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "allow" if allowed else "deny",
                    "permissionDecisionReason": reason,
                }
            }

        field_name = _TOOL_PATH_FIELDS.get(tool) if isinstance(tool, str) else None
        if field_name is None:
            return decision(False, f"tool {tool!r} is not permitted")
        raw = tool_input.get(field_name)
        if raw is None and tool in ("Glob", "Grep"):
            raw = "."  # these tools default to cwd
        if not isinstance(raw, str) or not raw:
            return decision(False, f"{tool} call missing {field_name}")
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = wt / candidate
        resolved = candidate.resolve()
        if tool in ("Read", "Glob", "Grep"):
            allowed = resolved == wt or resolved.is_relative_to(wt)
            scope = "the worktree"
        else:  # Write / Edit
            allowed = resolved == src or resolved.is_relative_to(src)
            scope = "worktree src/"
        return decision(allowed, f"{tool} {'inside' if allowed else 'outside'} "
                                 f"{scope}: {raw}")

    return guard


CODER_SYSTEM_PROMPT = (
    "You are a coding worker inside a constrained autoresearch loop. You "
    "implement exactly ONE scientific hypothesis by editing code in the "
    "current workspace, then stop. Hard constraints (violating tool calls "
    "are denied automatically):\n"
    "- Write/Edit only under src/ of this workspace; Read only inside this "
    "workspace. No Bash, no network, no new dependencies (Python stdlib "
    "only).\n"
    "- Keep training deterministic: fixed literal seeds only; never wall "
    "clock, os.urandom, or unseeded RNGs.\n"
    "- The entrypoint must remain `python src/train.py`, must honor "
    "AUTORESEARCH_SMOKE=1 (clamp to <=2 epochs), and must finish well "
    "within 90 seconds.\n"
    "- Preserve the `# --- HYPERPARAMS-BEGIN/END ---` marker block and the "
    "HYPERPARAMS dict literal in src/train.py (other experiments patch it "
    "mechanically).\n"
    "- artifacts/model.json must keep its schema: weights (finite floats), "
    "bias, hyperparams echo, feature_means/feature_stds (iff "
    "feature_scaling, matching the model input count), train_rmse, "
    "train_seconds, schema_version; optionally feature_spec — a list of "
    "engineered-feature terms, each a list of raw-feature indices whose "
    "PRODUCT forms one model input (max 32 terms, degree <= 3). weights "
    "length must equal the number of terms (8 raw features if no spec).\n"
    "- Implement only the given hypothesis; keep the diff minimal and "
    "atomic."
)


class ClaudeCoder:
    """Coding-agent executor over the Claude Agent SDK, confined to a
    worktree by the PreToolUse guard hook (see _make_worktree_guard)."""

    name = "coder"

    def __init__(self, contract: ResearchContract, model: str | None = None) -> None:
        self.contract = contract
        self.model = model

    def _options(self, worktree: Path, denials: list[dict],
                 resume: str | None = None):
        from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

        return ClaudeAgentOptions(
            cwd=str(worktree),
            setting_sources=[],
            tools=["Read", "Write", "Edit", "Glob", "Grep"],
            allowed_tools=[],  # nothing may shadow the guard hook
            disallowed_tools=["Bash", "WebFetch", "WebSearch", "Task",
                              "Skill", "TodoWrite", "NotebookEdit"],
            permission_mode="dontAsk",  # hook failure -> deny (fail closed)
            model=self.model,
            max_turns=self.contract.portfolio.coder_max_turns,
            max_budget_usd=self.contract.portfolio.coder_max_budget_usd,
            system_prompt=CODER_SYSTEM_PROMPT,
            resume=resume,
            hooks={"PreToolUse": [HookMatcher(
                matcher=None,
                hooks=[_make_worktree_guard(worktree, denials)],
                timeout=10,
            )]},
        )

    def _run(self, prompt: str, options) -> Any:
        import asyncio

        from claude_agent_sdk import ResultMessage, query

        async def go():
            result = None
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, ResultMessage):
                    result = message
            if result is None:
                raise CoderError("no result from coder agent")
            return result

        return asyncio.run(go())

    def _initial_prompt(self, hypothesis: Hypothesis, context: dict) -> str:
        return "\n\n".join([
            f"Hypothesis to implement:\n{json.dumps(hypothesis.to_dict(), indent=2)}",
            f"Implementation brief:\n{hypothesis.implementation_brief}",
            f"Incumbent dev metrics (for reference): "
            f"{json.dumps(context.get('dev_summary') or {})}",
            f"Recent distilled insights:\n"
            f"{json.dumps(context.get('insights') or [], indent=2)}",
            "The trainer is src/train.py (read it first). Acceptance: the "
            "smoke evaluation (AUTORESEARCH_SMOKE=1, 20s) and the dev "
            "evaluation (90s) must run your code to a valid "
            "artifacts/model.json. When done, summarize your change in a "
            "few sentences.",
        ])

    def execute(self, worktree: Path, hypothesis: Hypothesis,
                context: dict) -> dict:
        denials: list[dict] = []
        result = self._run(self._initial_prompt(hypothesis, context),
                           self._options(worktree, denials))
        if result.is_error:
            raise CoderError(f"coder agent errored: {result.result}")
        return {
            "summary": (result.result or "")[:500],
            "session_id": result.session_id,
            "cost_usd": result.total_cost_usd,
            "denied_tool_calls": denials,
        }

    def repair(self, worktree: Path, hypothesis: Hypothesis,
               session_id: str | None, failure_class: str,
               stderr_tail: str) -> dict:
        denials: list[dict] = []
        prompt = (
            f"The smoke evaluation of your implementation failed "
            f"mechanically ({failure_class}). stderr tail:\n"
            f"---\n{stderr_tail[-2000:]}\n---\n"
            f"Fix the mechanical problem only; do NOT change the scientific "
            f"intervention itself."
        )
        try:
            result = self._run(prompt, self._options(worktree, denials,
                                                     resume=session_id))
        except Exception:
            # Session resume can fail (persistence off / GC'd) — fall back to
            # a fresh call carrying the brief again.
            result = self._run(
                f"{self._initial_prompt(hypothesis, {})}\n\n{prompt}",
                self._options(worktree, denials),
            )
        if result.is_error:
            raise CoderError(f"coder repair errored: {result.result}")
        return {
            "summary": (result.result or "")[:500],
            "session_id": result.session_id,
            "cost_usd": result.total_cost_usd,
            "denied_tool_calls": denials,
        }


# ---------------------------------------------------------------------------
# Evaluator invocation
# ---------------------------------------------------------------------------

def run_evaluator(guard: ProtectionGuard, workspace: Path, mode: str,
                  out_path: Path, run_id: str, timeout_s: int,
                  split: str = "dev") -> dict:
    nonce = secrets.token_hex(16)
    cmd = [sys.executable, "-B", str(EVALUATOR_PATH),
           "--workspace", str(workspace), "--mode", mode, "--split", split,
           "--out", str(out_path), "--run-id", run_id, "--nonce", nonce]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout_s + 60)
    except subprocess.TimeoutExpired:
        raise EvaluatorInfraError(
            f"evaluator itself exceeded {timeout_s + 60}s — infra problem"
        ) from None
    if proc.returncode != 0:
        raise EvaluatorInfraError(
            f"evaluator exited {proc.returncode}: {proc.stderr[-2000:]}"
        )
    try:
        metrics = json.loads(out_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluatorInfraError(f"metrics unreadable: {exc}") from None

    if metrics.get("nonce") != nonce:
        raise ProtectionViolation(
            f"metrics nonce mismatch for {run_id}/{mode} — possible forgery"
        )
    if metrics.get("split") != split:
        raise ProtectionViolation(
            f"split echo mismatch for {run_id}: asked {split!r}, evaluator "
            f"reports {metrics.get('split')!r}"
        )
    manifest_files = guard.load_manifest()["files"]
    expected = manifest_files.get("evaluation/evaluate.py")
    actual = (metrics.get("evaluator") or {}).get("self_sha256")
    if expected and actual and expected != actual:
        raise ProtectionViolation(
            "evaluator self-hash does not match protection manifest"
        )
    # Metric identity check on EVERY evaluation: a contract edited to a
    # different name or direction must fail fast, not silently invert
    # classify()'s accept/reject semantics.
    reported = metrics.get("primary_metric") or {}
    contract_pm = guard.contract.primary_metric
    if (reported.get("name") != contract_pm.name
            or reported.get("direction") != contract_pm.direction):
        raise ContractError(
            f"primary-metric drift: evaluator reports "
            f"{reported.get('name')}/{reported.get('direction')} but the "
            f"contract expects {contract_pm.name}/{contract_pm.direction}"
        )
    return metrics


# ---------------------------------------------------------------------------
# State, classification, insights
# ---------------------------------------------------------------------------

def load_state() -> dict:
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise OrchestratorError(
            "no experiments/state.json — run `orchestrator.py init` first"
        ) from None
    except json.JSONDecodeError as exc:
        raise OrchestratorError(f"state.json corrupt: {exc}") from None
    version = state.get("schema_version")
    if version != STATE_SCHEMA_VERSION:
        raise OrchestratorError(
            f"state.json schema v{version} unsupported (this orchestrator "
            f"needs v{STATE_SCHEMA_VERSION}) — start a fresh campaign with "
            f"`orchestrator.py init --force`"
        )
    return state


def save_state(state: dict) -> None:
    atomic_write_json(STATE_PATH, state)


def classify(metrics: dict, best_primary: float | None,
             pm: PrimaryMetric) -> tuple[str, str | None, float | None]:
    """Returns (verdict, failure_class, primary_value)."""
    failure_class = metrics.get("failure_class")
    value = (metrics.get("primary_metric") or {}).get("value")
    if not metrics.get("executed") or metrics.get("degenerate") or value is None:
        return "valid_negative", failure_class, None
    value = float(value)
    if best_primary is None:
        return "valid_positive", None, value
    if pm.direction == "minimize":
        rel = (best_primary - value) / abs(best_primary)
    else:
        rel = (value - best_primary) / abs(best_primary)
    if rel >= pm.min_relative_improvement:
        return "valid_positive", None, value
    if rel <= -pm.min_relative_improvement:
        return "valid_negative", "metric_regression", value
    return "valid_inconclusive", None, value


def distill_insight(record: dict) -> dict | None:
    """Distill one proposer-visible lesson from an experiment record.

    BLINDNESS INVARIANT: this function must never read gate data — gate
    scores live only in record_type="gate" records (which return None at the
    first check below) and gate metrics files. An experiment's admission
    outcome is visible only as the accept/reject bit, which is inherently
    observable through incumbent movement anyway.
    """
    if record.get("record_type") != "experiment":
        return None
    hyp = record.get("hypothesis") or {}
    intervention = hyp.get("intervention") or {}
    param = intervention.get("param")
    frm, to = intervention.get("from"), intervention.get("to")
    verdict = record.get("verdict")
    decision = record.get("decision")
    fc = record.get("failure_class")
    before = record.get("best_primary_before")
    primary = record.get("primary")

    before_txt = f"{before:.4f}" if isinstance(before, (int, float)) else "?"
    primary_txt = f"{primary:.4f}" if isinstance(primary, (int, float)) else "n/a"
    if param is not None:
        desc = f"{param} {frm!r}->{to!r}"
    else:
        desc = f"coder: {(hyp.get('statement') or 'code intervention')[:80]}"

    if verdict == "valid_positive" and decision == "accept":
        observation = (f"{desc} improved heldout_rmse "
                       f"{before_txt}->{primary_txt} and was admitted.")
        follow_up = (f"continue moving {param} in this direction"
                     if param is not None else
                     "build on this code change in later hypotheses")
        confidence = 0.8
    elif verdict == "valid_positive":
        observation = (f"{desc} improved the dev split "
                       f"{before_txt}->{primary_txt} but was NOT admitted by "
                       f"the blind gate this generation.")
        follow_up = ("dev-split gains of this size may not generalize; "
                     "prefer interventions with a stronger mechanism")
        confidence = 0.6
    elif fc in ("degenerate_weights", "no_skill"):
        observation = (f"{desc} made training degenerate ({fc}); "
                       f"optimization left the stable region.")
        follow_up = (f"treat {to!r} as an upper bound for {param} in this regime"
                     if param is not None else
                     "this code direction destabilizes training")
        confidence = 0.9
    elif fc in ("timeout", "nonzero_exit"):
        observation = (f"{desc} made the run fail ({fc}) despite a valid "
                       f"implementation — infeasible under the budget.")
        follow_up = (f"avoid pushing {param} further this way"
                     if param is not None else
                     "keep code changes within the training budget")
        confidence = 0.7
    elif verdict == "valid_negative":
        observation = (f"{desc} regressed heldout_rmse "
                       f"{before_txt}->{primary_txt}; rejected.")
        follow_up = (f"deprioritize this direction for {param}"
                     if param is not None else
                     "deprioritize this code direction")
        confidence = 0.7
    elif verdict == "valid_inconclusive":
        observation = (f"{desc} changed heldout_rmse "
                       f"{before_txt}->{primary_txt}, below the "
                       f"min_relative_improvement threshold.")
        follow_up = "not the binding constraint near this optimum"
        confidence = 0.5
    elif verdict in ("invalid_implementation", "contract_violation",
                     "ff_conflict", "aborted"):
        observation = (f"round {record.get('run_id')} ended as {verdict} "
                       f"({fc or 'no failure class'}) — no scientific signal.")
        follow_up = "no scientific conclusion; mechanical/protocol issue"
        confidence = 0.3
    else:
        return None

    return {
        "insight_id": f"ins_{record.get('run_id', '?')}",
        "scope": "synthetic heteroscale regression (hyperparams + src/** code)",
        "round": record.get("round"),
        "generation": record.get("generation"),
        "hypothesis_id": hyp.get("id"),
        "observation": observation,
        "outcome": verdict,
        "failure_class": fc,
        "conditions": {"param": param, "from": frm, "to": to,
                       "executor": record.get("executor", "patcher")},
        "recommended_follow_up": follow_up,
        "supporting_run_ids": [record.get("run_id")],
        "confidence": confidence,
    }


def corrected_run_ids(records: list[dict]) -> set:
    return {r.get("corrects") for r in records
            if r.get("record_type") == "correction"}


def rebuild_insights() -> list[dict]:
    """insight_memory.json is derived data: always rebuildable from the ledger."""
    records = read_jsonl(LEDGER_PATH)
    corrected = corrected_run_ids(records)
    insights = []
    for record in records:
        if record.get("run_id") in corrected:
            continue  # accept that failed to merge: no scientific standing
        insight = distill_insight(record)
        if insight:
            insights.append(insight)
    atomic_write_json(INSIGHTS_PATH, insights)
    return insights


def replay_ledger_fields(state: dict, records: list[dict]) -> None:
    """Rebuild tested/stagnation/last_accepted from the ledger.

    These fields are mutated in memory between the write-ahead ledger append
    and save_state; a crash in that window would otherwise lose them (wrong
    stagnation stop, re-proposal of already-measured interventions). They are
    pure functions of the ledger, so recovery recomputes them wholesale.
    """
    corrected = corrected_run_ids(records)
    tested: dict[str, list[str]] = {}
    max_generation = 0

    # Stagnation counts GENERATIONS without an admitted winner, so records
    # must be grouped by generation before replay. Legacy records without a
    # generation field each form a singleton group in file order (exact
    # Phase 1 semantics).
    groups: list[list[dict]] = []
    group_index: dict[int, int] = {}
    for r in records:
        if r.get("record_type") != "experiment" or r.get("verdict") == "aborted":
            continue
        intervention = (r.get("hypothesis") or {}).get("intervention") or {}
        param = intervention.get("param")
        if param is not None:
            values = tested.setdefault(param, [])
            for v in (intervention.get("to"), intervention.get("from")):
                vr = value_repr(v)
                if vr not in values:
                    values.append(vr)
        generation = r.get("generation")
        if isinstance(generation, int):
            max_generation = max(max_generation, generation)
            if generation not in group_index:
                group_index[generation] = len(groups)
                groups.append([])
            groups[group_index[generation]].append(r)
        else:
            groups.append([r])

    stagnation = 0
    last_accepted = None
    for group in groups:
        winner = next(
            (r for r in group
             if r.get("decision") == "accept"
             and r.get("run_id") not in corrected),
            None,
        )
        if winner is not None:
            stagnation = 0
            last_accepted = dict(
                (winner.get("hypothesis") or {}).get("intervention") or {}
            )
        else:
            stagnation += 1

    state["tested"] = tested
    state["stagnation"] = stagnation
    state["last_accepted"] = last_accepted
    state["generation"] = max(state.get("generation") or 0, max_generation)


# ---------------------------------------------------------------------------
# Generation execution (parallel hypothesis portfolio)
# ---------------------------------------------------------------------------

@dataclass
class LoopContext:
    contract: ResearchContract
    git: Git
    guard: ProtectionGuard
    patcher: HyperparamsPatcher
    proposer: Any
    coder: ClaudeCoder | None = None


@dataclass
class ExperimentSpec:
    run_id: str
    generation: int
    hypothesis: Hypothesis
    branch: str
    worktree: Path
    round_dir: Path
    base_commit: str
    insights: list[dict] = field(default_factory=list)


# Smoke-stage failure classes meaning "no scoreable model was ever produced"
# — the coder-era analogue of PatchError, and the ONLY repairable outcomes.
# timeout / degenerate_weights / no_skill / any dev-stage failure are
# scientific evidence and are never repaired (false-repair prevention).
MECHANICAL_FAILURES = {"nonzero_exit", "missing_artifact", "malformed_artifact"}

MAX_DIFF_BYTES = 200_000


def _glob_violations(ctx: LoopContext, spec: "ExperimentSpec") -> list[str]:
    paths = set(ctx.git.diff_paths(spec.base_commit, cwd=spec.worktree))
    paths.update(ctx.git.status_paths(cwd=spec.worktree))
    return [p for p in sorted(paths)
            if matches_any(p, ctx.contract.protected_globs)
            or not matches_any(p, ctx.contract.editable_globs)]


def _root_fingerprint(ctx: LoopContext) -> tuple:
    """Snapshot of the ROOT checkout used to detect coder escape. Combines
    git status (tracked + untracked) with the protection manifest so a rogue
    absolute-path write to ROOT/src/** (editable, so NOT in the manifest) is
    still caught as a status change."""
    return (tuple(ctx.git.status_paths()), tuple(ctx.guard.verify()))


def _assert_root_unchanged(run_id: str, before: tuple, after: tuple) -> None:
    """After any coder invocation: cwd does not confine absolute-path tool
    calls, so a mutation of the ROOT checkout means containment failed and
    every later round's baseline is suspect. Compare against the pre-call
    snapshot (root may legitimately carry pre-existing untracked files) and
    halt the campaign on any coder-introduced change."""
    if after != before:
        new_status = set(after[0]) - set(before[0])
        new_viol = set(after[1]) - set(before[1])
        raise ProtectionViolation(
            f"root working tree mutated during coder round {run_id}: "
            f"status={sorted(new_status)} protection={sorted(new_viol)}"
        )


def _scan_src_symlinks(worktree: Path) -> list[str]:
    """A symlink under src/ passes the editable-glob check by path while
    redirecting content elsewhere; reject outright."""
    bad = []
    for dirpath, dirnames, filenames in os.walk(worktree / "src"):
        for name in dirnames + filenames:
            p = Path(dirpath) / name
            if p.is_symlink():
                bad.append(p.relative_to(worktree).as_posix())
    return bad


def _commit_worktree(ctx: LoopContext, spec: "ExperimentSpec",
                     message: str) -> tuple[str | None, str | None]:
    """Returns (commit, failure); failure='empty_diff' when nothing changed."""
    ctx.git.run("add", "-A", cwd=spec.worktree)
    proc = ctx.git.run("commit", "-m", message, cwd=spec.worktree, check=False)
    if proc.returncode != 0:
        output = proc.stdout + proc.stderr
        if "nothing to commit" in output or "nothing added to commit" in output:
            return None, "empty_diff"
        raise GitError(f"worktree commit failed: {output.strip()}")
    return ctx.git.head(cwd=spec.worktree), None


def _finish_generation(ctx: LoopContext, state: dict,
                       specs: list["ExperimentSpec"], records: list[dict],
                       gate_record: dict, winner_run_id: str | None) -> dict:
    """Merge-guard preflight, write-ahead ledger batch (gate first), merge,
    state update, cleanup — the single persistence path per generation."""
    winner = next((r for r in records if r["run_id"] == winner_run_id), None)

    if winner is not None:
        # Pre-flight BEFORE the write-ahead batch: an accept that cannot land
        # (main moved/dirty mid-generation) must never enter the ledger as
        # accepted — it would poison insights and replay forever.
        head = ctx.git.head()
        dirty = ctx.git.status_paths(include_untracked=False)
        if head != winner["base_commit"] or dirty:
            winner["verdict"] = "ff_conflict"
            winner["failure_class"] = (
                f"main moved during generation "
                f"({winner['base_commit'][:12]} -> {head[:12]})"
                if head != winner["base_commit"]
                else f"main working tree dirty: {dirty}"
            )
            gate_record["winner"] = None
            gate_record["reason"] = (gate_record.get("reason") or "") + \
                " (demoted: merge guard failed)"
            winner = None

    for r in records:
        r["decision"] = ("accept" if winner is not None and r is winner
                         else "reject")
        r["best_primary_after"] = (
            r["primary"] if r["decision"] == "accept"
            else state.get("best_primary")
        )

    # Write-ahead batch: the gate record FIRST, so any accept record that
    # exists always has its admission provenance; then experiments in run
    # order. Recovery keys off experiment decisions only.
    append_jsonl(LEDGER_PATH, gate_record)
    for r in records:
        append_jsonl(LEDGER_PATH, r)

    if winner is not None:
        try:
            ctx.git.merge_ff(winner["commit"])
        except GitError as exc:
            append_jsonl(LEDGER_PATH, {
                "record_type": "correction",
                "corrects": winner["run_id"],
                "timestamp_utc": utc_now(),
                "reason": f"ff-merge failed after accept was recorded: {exc}",
            })
            raise
        # best_primary is the DEV score — never the gate score, which must
        # not flow into proposer-visible state.
        state["best_primary"] = winner["primary"]
        state["best_commit"] = winner["commit"]
        state["stagnation"] = 0
        state["last_accepted"] = dict(winner["hypothesis"]["intervention"])
        winner_gate = (gate_record.get("results") or {}).get(winner["run_id"])
        if winner_gate is not None:
            state.setdefault("gate", {}).setdefault("incumbent_scores", {})[
                winner["commit"]] = winner_gate
    else:
        state["stagnation"] = state.get("stagnation", 0) + 1

    # Register both endpoints as tested for every completed hyperparameter
    # experiment (winner and losers alike, including gate-rejected ones).
    for r in records:
        if r.get("verdict") == "aborted":
            continue
        intervention = (r.get("hypothesis") or {}).get("intervention") or {}
        param = intervention.get("param")
        if param is None:
            continue  # coder hypotheses have no single-param endpoint
        values = state.setdefault("tested", {}).setdefault(param, [])
        for v in (value_repr(intervention.get("to")),
                  value_repr(intervention.get("from"))):
            if v not in values:
                values.append(v)

    state["generation"] = gate_record["generation"]

    # Worktrees BEFORE branch deletion: git refuses to delete a branch that
    # is checked out in a live worktree.
    for spec in specs:
        ctx.git.worktree_remove(spec.worktree)
    for spec, r in zip(specs, records):
        if (r.get("verdict") == "invalid_implementation"
                and ctx.git.branch_exists(spec.branch)):
            ctx.git.delete_branch(spec.branch)

    rebuild_insights()
    state["current_generation"] = None
    save_state(state)
    return {"records": records, "gate": gate_record,
            "winner": winner["run_id"] if winner is not None else None}


def run_experiment(ctx: LoopContext, spec: ExperimentSpec,
                   best_primary_snapshot: float | None) -> dict:
    """Worker-thread body for one hypothesis. Touches ONLY its own worktree
    and round dir; all state/ledger persistence happens on the main thread
    after the generation barrier. Classification compares against the
    generation-start incumbent snapshot so all K see the same target."""
    contract = ctx.contract
    pm = contract.primary_metric
    hyp = spec.hypothesis

    record: dict = {
        "record_type": "experiment",
        "run_id": spec.run_id,
        "round": int(spec.run_id[1:]),
        "generation": spec.generation,
        "timestamp_utc": utc_now(),
        "hypothesis": hyp.to_dict(),
        "executor": hyp.executor,
        "branch": spec.branch,
        "base_commit": spec.base_commit,
        "commit": None,
        "verdict": None,
        "failure_class": None,
        "decision": "reject",
        "primary": None,
        "best_primary_before": best_primary_snapshot,
        "metrics_path": None,
        "proposer": hyp.proposer,
    }

    # --- implement ------------------------------------------------------------
    session_id: str | None = None
    if hyp.executor == "coder":
        if ctx.coder is None:
            record.update(verdict="invalid_implementation",
                          failure_class="coder_unavailable")
            return record
        root_before = _root_fingerprint(ctx)
        try:
            coder_result = ctx.coder.execute(
                spec.worktree, hyp, {"insights": spec.insights},
            )
        except CoderError as exc:
            _assert_root_unchanged(spec.run_id, root_before,
                                   _root_fingerprint(ctx))
            record.update(verdict="invalid_implementation",
                          failure_class=f"coder_error: {exc}")
            return record
        _assert_root_unchanged(spec.run_id, root_before, _root_fingerprint(ctx))
        session_id = coder_result.get("session_id")
        record["coder"] = {
            "cost_usd": coder_result.get("cost_usd"),
            "summary": coder_result.get("summary"),
            "repairs": 0,
            "denied_tool_calls": len(coder_result.get("denied_tool_calls") or []),
        }
    else:
        param = hyp.intervention["param"]
        new_value = hyp.intervention["to"]
        train_file = spec.worktree / TRAIN_REL
        patch_error: PatchError | None = None
        for attempt in range(contract.budgets.repair_attempts + 1):
            try:
                ctx.patcher.apply(train_file, param, new_value, attempt=attempt)
                patch_error = None
                break
            except PatchError as exc:
                patch_error = exc
                ctx.git.run("checkout", "--", TRAIN_REL, cwd=spec.worktree,
                            check=False)
        if patch_error is not None:
            record.update(verdict="invalid_implementation",
                          failure_class=f"patch_failed: {patch_error}")
            return record

    bad_links = _scan_src_symlinks(spec.worktree)
    if bad_links:
        record.update(verdict="contract_violation",
                      failure_class=f"symlink under src/: {bad_links}")
        return record

    commit, failure = _commit_worktree(
        ctx, spec, f"{spec.run_id}: {hyp.statement[:100]}\n\n"
                   f"Hypothesis: {hyp.statement}",
    )
    if failure:
        record.update(verdict="invalid_implementation", failure_class=failure)
        return record
    record["commit"] = commit

    diff_bytes = len(ctx.git.run("diff", f"{spec.base_commit}..HEAD",
                                 cwd=spec.worktree).stdout.encode())
    if diff_bytes > MAX_DIFF_BYTES:
        record.update(verdict="invalid_implementation",
                      failure_class=f"oversized_diff: {diff_bytes} bytes")
        return record

    bad = _glob_violations(ctx, spec)
    if bad:
        record.update(verdict="contract_violation",
                      failure_class=f"pre-eval protected/editable violation: {bad}")
        return record

    # --- smoke, with bounded coder repair for MECHANICAL failures only --------
    smoke_path = spec.round_dir / "metrics_smoke.json"
    smoke = run_evaluator(ctx.guard, spec.worktree, "smoke", smoke_path,
                          spec.run_id, contract.budgets.smoke_train_timeout_s)
    record["smoke_metrics_path"] = str(smoke_path)

    repairs = 0
    while (hyp.executor == "coder" and ctx.coder is not None
           and smoke.get("failure_class") in MECHANICAL_FAILURES
           and repairs < contract.budgets.repair_attempts):
        repairs += 1
        root_before = _root_fingerprint(ctx)
        try:
            repair_result = ctx.coder.repair(
                spec.worktree, hyp, session_id,
                str(smoke.get("failure_class")),
                str(smoke.get("stderr_tail") or ""),
            )
        except CoderError as exc:
            record["failure_class"] = f"coder_repair_error: {exc}"
            break
        _assert_root_unchanged(spec.run_id, root_before, _root_fingerprint(ctx))
        session_id = repair_result.get("session_id") or session_id
        record["coder"]["repairs"] = repairs
        record["coder"]["denied_tool_calls"] += len(
            repair_result.get("denied_tool_calls") or [])
        bad_links = _scan_src_symlinks(spec.worktree)
        if bad_links:
            record.update(verdict="contract_violation",
                          failure_class=f"symlink under src/: {bad_links}")
            return record
        commit, failure = _commit_worktree(
            ctx, spec, f"{spec.run_id}: mechanical repair attempt {repairs}")
        if failure is None:
            record["commit"] = commit
        bad = _glob_violations(ctx, spec)
        if bad:
            record.update(verdict="contract_violation",
                          failure_class=f"post-repair violation: {bad}")
            return record
        smoke = run_evaluator(ctx.guard, spec.worktree, "smoke", smoke_path,
                              spec.run_id,
                              contract.budgets.smoke_train_timeout_s)

    if not smoke.get("executed") or smoke.get("degenerate"):
        if (hyp.executor == "coder"
                and smoke.get("failure_class") in MECHANICAL_FAILURES):
            # Repair budget exhausted without ever producing a scoreable
            # model: mechanical, not scientific.
            record.update(verdict="invalid_implementation",
                          failure_class=f"smoke_{smoke.get('failure_class')}",
                          metrics_path=str(smoke_path))
        else:
            verdict, failure_class, primary = classify(
                smoke, best_primary_snapshot, pm)
            record.update(verdict=verdict, failure_class=failure_class,
                          primary=primary, metrics_path=str(smoke_path))
        return record

    # --- dev --------------------------------------------------------------------
    dev_path = spec.round_dir / "metrics_dev.json"
    dev = run_evaluator(ctx.guard, spec.worktree, "dev", dev_path,
                        spec.run_id, contract.budgets.dev_train_timeout_s)
    verdict, failure_class, primary = classify(dev, best_primary_snapshot, pm)
    record.update(verdict=verdict, failure_class=failure_class,
                  primary=primary, metrics_path=str(dev_path))

    # --- post-evaluation tamper check --------------------------------------------
    bad = _glob_violations(ctx, spec)
    if bad:
        record.update(verdict="contract_violation",
                      failure_class=f"post-eval protected/editable violation: {bad}")
    return record


def _run_gate(ctx: LoopContext, state: dict, records: list[dict],
              generation: int, base_commit: str) -> tuple[dict, str | None]:
    """Blind admission gate over the generation's dev improvers.

    Gate scores live ONLY in this record and the gate metrics files — never
    in experiment records, insights, or proposer context (blindness by
    construction: everything proposer-visible is derived from experiment
    records, which carry no gate fields)."""
    contract = ctx.contract
    pm = contract.primary_metric
    pf = contract.portfolio

    gate_record: dict = {
        "record_type": "gate",
        "generation": generation,
        "timestamp_utc": utc_now(),
        "base_commit": base_commit,
        "candidates": [],
        "incumbent_gate": None,
        "incumbent_evaluated": False,
        "results": {},
        "winner": None,
        "reason": None,
        "selection_rule": (
            f"winner = best gate-split improver over the incumbent's gate "
            f"score by >= {pf.gate_min_relative_improvement:.2%} relative, "
            f"among the top {pf.gate_top_k} dev improvers"
        ),
    }

    improvers = [r for r in records if r.get("verdict") == "valid_positive"
                 and isinstance(r.get("primary"), float)]
    improvers.sort(key=lambda r: r["primary"],
                   reverse=pm.direction == "maximize")
    candidates = improvers[: pf.gate_top_k]
    gate_record["candidates"] = [r["run_id"] for r in candidates]
    if not candidates:
        gate_record["reason"] = "no dev improvers"
        return gate_record, None

    gate_dir = EXPERIMENTS_DIR / "generations" / f"g{generation:04d}" / "gate"

    # Determinism recheck for gate-bound candidates only: coder-authored code
    # can smuggle nondeterminism past PYTHONHASHSEED, and the epsilon
    # keep-rule's justification depends on (config -> score) being pure.
    verified = []
    for r in candidates:
        recheck = run_evaluator(
            ctx.guard, WORKTREES_DIR / r["run_id"], "dev",
            gate_dir / f"{r['run_id']}_dev_recheck.json", r["run_id"],
            contract.budgets.dev_train_timeout_s,
        )
        if (recheck.get("primary_metric") or {}).get("value") != r["primary"]:
            r["verdict"] = "invalid_implementation"
            r["failure_class"] = "nondeterministic"
            r["primary"] = None
            continue
        verified.append(r)
    if not verified:
        gate_record["reason"] = "all candidates failed the determinism recheck"
        return gate_record, None

    # Incumbent gate score, cached per commit. Evaluation is a pure function
    # of (commit, split seed, evaluator hash), so the cache is exact, not an
    # approximation. ROOT sits at base_commit and clean here (cmd_run's dirty
    # check + nothing merges mid-generation).
    cache = state.setdefault("gate", {}).setdefault("incumbent_scores", {})
    incumbent_gate = cache.get(base_commit)
    if incumbent_gate is None:
        metrics = run_evaluator(ctx.guard, ROOT, "dev",
                                gate_dir / "incumbent.json",
                                f"g{generation:04d}-incumbent",
                                contract.budgets.dev_train_timeout_s,
                                split="gate")
        incumbent_gate = (metrics.get("primary_metric") or {}).get("value")
        if incumbent_gate is None:
            raise EvaluatorInfraError("incumbent gate evaluation failed")
        cache[base_commit] = incumbent_gate
        gate_record["incumbent_evaluated"] = True
    gate_record["incumbent_gate"] = incumbent_gate

    best_record: dict | None = None
    best_value: float | None = None
    for r in verified:
        metrics = run_evaluator(ctx.guard, WORKTREES_DIR / r["run_id"], "dev",
                                gate_dir / f"{r['run_id']}.json", r["run_id"],
                                contract.budgets.dev_train_timeout_s,
                                split="gate")
        value = (metrics.get("primary_metric") or {}).get("value")
        gate_record["results"][r["run_id"]] = value
        if value is None:
            continue
        if pm.direction == "minimize":
            rel = (incumbent_gate - value) / abs(incumbent_gate)
            better = best_value is None or value < best_value
        else:
            rel = (value - incumbent_gate) / abs(incumbent_gate)
            better = best_value is None or value > best_value
        if rel >= pf.gate_min_relative_improvement and better:
            best_record, best_value = r, value

    if best_record is not None:
        gate_record["winner"] = best_record["run_id"]
        gate_record["reason"] = "beat the incumbent on the blind gate split"
    else:
        gate_record["reason"] = "no candidate beat the incumbent on the gate split"
    return gate_record, gate_record["winner"]


def run_generation(ctx: LoopContext, state: dict) -> dict | None:
    """Execute one parallel portfolio generation. Returns a summary dict
    ({"stop": reason} for budget stops), or None when the proposer's move
    space is exhausted."""
    contract = ctx.contract
    pf = contract.portfolio

    violations = ctx.guard.verify()
    if violations:
        raise ProtectionViolation("; ".join(violations))

    k_budget = min(pf.parallel_branches,
                   contract.budgets.max_rounds - state["round"])
    if k_budget <= 0:
        return {"stop": "max_rounds"}

    generation = state.get("generation", 0) + 1
    insights = read_insights()
    batch = ctx.proposer.propose_batch(ProposalContext(
        contract=contract,
        round_index=state["round"] + 1,
        current_hyperparams=ctx.patcher.read(ROOT / TRAIN_REL),
        best_primary=state.get("best_primary"),
        tested=state.get("tested", {}),
        last_accepted=state.get("last_accepted"),
        insights=insights,
    ), k_budget)
    if not batch:
        return None

    campaign = state.get("campaign_id", "c0")
    base_commit = ctx.git.head()
    specs: list[ExperimentSpec] = []
    for i, hyp in enumerate(batch):
        rnum = state["round"] + 1 + i
        run_id = f"r{rnum:04d}"
        hyp.round = rnum
        hyp.id = f"h_{run_id}_{_slug(hyp)}"
        specs.append(ExperimentSpec(
            run_id=run_id,
            generation=generation,
            hypothesis=hyp,
            branch=f"hyp/{campaign}/{run_id}-{_slug(hyp)}",
            worktree=WORKTREES_DIR / run_id,
            round_dir=ROUNDS_DIR / run_id,
            base_commit=base_commit,
            insights=insights[-5:],
        ))

    # Write-ahead point 1: reserve the contiguous rNNNN block + generation
    # marker BEFORE any git work; round numbers are never reused.
    state["round"] += len(specs)
    state["current_generation"] = {
        "generation": generation,
        "base_commit": base_commit,
        "phase": "started",
        "started_utc": utc_now(),
        "experiments": [{"run_id": s.run_id, "branch": s.branch,
                         "worktree": str(s.worktree)} for s in specs],
    }
    save_state(state)
    for spec in specs:
        spec.round_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(spec.round_dir / "hypothesis.json",
                          spec.hypothesis.to_dict())

    # Serial worktree creation: worktree add mutates shared .git admin state.
    for spec in specs:
        if ctx.git.branch_exists(spec.branch):
            raise GitError(f"branch {spec.branch} already exists — "
                           f"recovery did not run?")
        ctx.git.worktree_add(spec.worktree, spec.branch, base_commit)

    state["current_generation"]["phase"] = "executing"
    save_state(state)

    snapshot = state.get("best_primary")
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=len(specs)) as pool:
        futures = {pool.submit(run_experiment, ctx, spec, snapshot): spec
                   for spec in specs}
        try:
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    results[spec.run_id] = future.result()
                except ProtectionViolation:
                    raise  # tamper is never a per-branch condition
                except EvaluatorInfraError as exc:
                    # One flaky evaluation must not burn the generation.
                    results[spec.run_id] = {
                        "record_type": "experiment",
                        "run_id": spec.run_id,
                        "round": int(spec.run_id[1:]),
                        "generation": generation,
                        "timestamp_utc": utc_now(),
                        "hypothesis": spec.hypothesis.to_dict(),
                        "executor": spec.hypothesis.executor,
                        "branch": spec.branch,
                        "base_commit": base_commit,
                        "commit": None,
                        "verdict": "aborted",
                        "failure_class": f"evaluator_infra: {exc}",
                        "decision": "reject",
                        "primary": None,
                        "best_primary_before": snapshot,
                        "proposer": spec.hypothesis.proposer,
                    }
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise

    ordered = [results[s.run_id] for s in specs]
    if getattr(ctx.proposer, "last_cost_usd", None) is not None:
        # One batch call funded the whole generation; attribute to the first.
        ordered[0]["proposal_cost_usd"] = ctx.proposer.last_cost_usd

    state["current_generation"]["phase"] = "evaluated"
    save_state(state)

    gate_record, winner_run_id = _run_gate(ctx, state, ordered, generation,
                                           base_commit)
    state["current_generation"]["phase"] = "gated"
    save_state(state)

    return _finish_generation(ctx, state, specs, ordered, gate_record,
                              winner_run_id)


def read_insights() -> list[dict]:
    try:
        data = json.loads(INSIGHTS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        # Derived data: rebuild from the ledger instead of failing.
        return rebuild_insights()


# ---------------------------------------------------------------------------
# Crash recovery
# ---------------------------------------------------------------------------

def recover(ctx: LoopContext) -> None:
    if not STATE_PATH.exists():
        return
    state = load_state()
    git = ctx.git

    git.worktree_prune()
    if WORKTREES_DIR.exists():
        for child in sorted(WORKTREES_DIR.iterdir()):
            if child.is_dir():
                print(f"[recover] removing stale worktree {child.name}",
                      file=sys.stderr)
                git.worktree_remove(child)
        git.worktree_prune()

    current = state.get("current_generation")
    if current:
        generation = current.get("generation")
        base = current.get("base_commit")
        ledger_ids = {r.get("run_id") for r in read_jsonl(LEDGER_PATH)}
        for exp in current.get("experiments", []):
            run_id = exp.get("run_id")
            branch = exp.get("branch")
            if branch and git.branch_exists(branch) \
                    and git.branch_tip(branch) == base:
                git.delete_branch(branch)  # no commit: nothing to preserve
            if run_id in ledger_ids:
                # A crash after the write-ahead batch: real records exist;
                # never overwrite them with aborted ones.
                continue
            print(f"[recover] round {run_id} (generation {generation}) was "
                  f"interrupted; marking aborted", file=sys.stderr)
            aborted_record = {
                "record_type": "experiment",
                "run_id": run_id,
                "round": int(str(run_id)[1:]) if str(run_id)[1:].isdigit()
                else None,
                "generation": generation,
                "timestamp_utc": utc_now(),
                "verdict": "aborted",
                "failure_class": "interrupted",
                "decision": "reject",
                "branch": branch,
                "base_commit": base,
                "commit": None,
                "best_primary_before": state.get("best_primary"),
            }
            # The hypothesis certificate is persisted to the round dir before
            # any git work, so an interrupted round can usually still be
            # attributed (insights, status display).
            try:
                aborted_record["hypothesis"] = json.loads(
                    (ROUNDS_DIR / str(run_id) / "hypothesis.json")
                    .read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                pass
            append_jsonl(LEDGER_PATH, aborted_record)
        state["current_generation"] = None

    # Replay accepted-but-unmerged rounds (ledger verdict is the write-ahead log).
    records = read_jsonl(LEDGER_PATH)
    corrected = corrected_run_ids(records)
    accepted = [r for r in records
                if r.get("decision") == "accept" and r.get("commit")
                and r.get("run_id") not in corrected]
    if accepted:
        last = accepted[-1]
        head = git.head()
        if head != last["commit"]:
            if git.is_ancestor(head, last["commit"]):
                print(f"[recover] replaying accepted merge {last['run_id']}",
                      file=sys.stderr)
                dirty = git.status_paths(include_untracked=False)
                if dirty:
                    raise GitError(f"cannot replay merge; main dirty: {dirty}")
                git.merge_ff(last["commit"])
            else:
                # main diverged (e.g. a manual commit) — the accept can never
                # land. Record a correction so ledger, insights, and replayed
                # state stop claiming an improvement that main never got.
                print(f"[recover] accepted merge {last['run_id']} cannot be "
                      f"replayed (main diverged); recording correction",
                      file=sys.stderr)
                append_jsonl(LEDGER_PATH, {
                    "record_type": "correction",
                    "corrects": last["run_id"],
                    "timestamp_utc": utc_now(),
                    "reason": "accepted merge could not be replayed: main "
                              "diverged from the hypothesis commit",
                })
        if git.head() == last["commit"] and state.get("best_commit") != last["commit"]:
            state["best_commit"] = last["commit"]
            state["best_primary"] = last.get("primary")

    # Ledger-derived fields survive any crash window (see replay_ledger_fields).
    # Re-read: corrections may have been appended above.
    replay_ledger_fields(state, read_jsonl(LEDGER_PATH))
    save_state(state)
    rebuild_insights()


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def _load_evaluator_declarations() -> dict:
    spec = importlib.util.spec_from_file_location("autoresearch_evaluator",
                                                  EVALUATOR_PATH)
    if spec is None or spec.loader is None:
        raise OrchestratorError("cannot import evaluation/evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {
        "budgets": dict(module.TRAIN_TIMEOUT_S),
        "metric_name": module.PRIMARY_METRIC_NAME,
        "metric_direction": module.PRIMARY_METRIC_DIRECTION,
        "split_names": tuple(module.SPLIT_NAMES),
    }


def cmd_init(args: argparse.Namespace) -> int:
    contract = load_contract()
    git = Git(ROOT)
    guard = ProtectionGuard(ROOT, contract)

    if STATE_PATH.exists() and not args.force:
        raise OrchestratorError(
            "already initialized (experiments/state.json exists); "
            "use --force to re-baseline"
        )

    required = [CONTRACT_PATH, EVALUATOR_PATH, ROOT / "evaluation" / "dataset.py",
                ROOT / TRAIN_REL, ROOT / ".gitignore", ROOT / "pyproject.toml"]
    for path in required:
        if not path.is_file():
            raise OrchestratorError(f"required file missing: {path}")

    declared = _load_evaluator_declarations()
    if (declared["budgets"].get("smoke") != contract.budgets.smoke_train_timeout_s
            or declared["budgets"].get("dev") != contract.budgets.dev_train_timeout_s):
        raise ContractError(
            f"budget drift: contract says smoke/dev = "
            f"{contract.budgets.smoke_train_timeout_s}/"
            f"{contract.budgets.dev_train_timeout_s}s but evaluator hardcodes "
            f"{declared['budgets']}"
        )
    if (declared["metric_name"] != contract.primary_metric.name
            or declared["metric_direction"] != contract.primary_metric.direction):
        raise ContractError(
            f"metric drift: contract says "
            f"{contract.primary_metric.name}/{contract.primary_metric.direction} "
            f"but evaluator hardcodes "
            f"{declared['metric_name']}/{declared['metric_direction']}"
        )
    if declared["split_names"] != ("dev", "gate", "test"):
        raise ContractError(
            f"split drift: this orchestrator expects dev/gate/test but the "
            f"evaluator declares {declared['split_names']}"
        )

    if not git.is_repo():
        git.init_repo()
        print("[init] git repository created (branch: main)")
    # A background `gc --auto` (spawned by merges) could rewrite packed-refs
    # while parallel worktree operations run; disable it outright.
    git.run("config", "gc.auto", "0")

    if args.force and EXPERIMENTS_DIR.exists():
        shutil.rmtree(EXPERIMENTS_DIR)
        print("[init] --force: cleared experiments/")

    if not HELDOUT_CONFIG_PATH.exists() or args.force:
        seeds = set()
        while len(seeds) < 3:  # pairwise-distinct hidden seeds
            seeds.add(int.from_bytes(os.urandom(8), "big") % (2**31 - 1))
        dev_seed, gate_seed, test_seed = sorted(seeds)
        atomic_write_json(HELDOUT_CONFIG_PATH, {
            "schema_version": 2,
            "splits": {
                "dev": {"seed": dev_seed},
                "gate": {"seed": gate_seed},
                "test": {"seed": test_seed},
            },
            "created_utc": utc_now(),
            "note": "hidden held-out seeds (dev=search, gate=blind admission, "
                    "test=final report); untracked by git on purpose",
        })
        print("[init] generated evaluation/heldout_config.json "
              "(hidden dev/gate/test seeds)")

    guard.write_manifest()
    print(f"[init] protection manifest written "
          f"({len(guard.load_manifest()['files'])} files)")

    if not git.has_head():
        git.commit_all("AutoResearch scaffold\n\n"
                       "Contract, evaluator, dataset, trainer, orchestrator, "
                       "protection manifest.")
        print(f"[init] initial commit {git.head()[:12]}")
    elif git.status_paths(include_untracked=False):
        git.commit_all("Rebuild protection manifest")
        print(f"[init] committed manifest update {git.head()[:12]}")

    print("[init] running baseline evaluation ...")
    metrics = run_evaluator(guard, ROOT, "dev", BASELINE_DIR / "metrics_dev.json",
                            "baseline", contract.budgets.dev_train_timeout_s)
    value = (metrics.get("primary_metric") or {}).get("value")
    if not metrics.get("executed") or value is None:
        raise OrchestratorError(
            f"baseline evaluation must succeed (failure_class="
            f"{metrics.get('failure_class')}); it anchors the attribution rule"
        )

    violations = guard.verify()
    if violations:
        raise ProtectionViolation(
            "baseline evaluation touched protected files: " + "; ".join(violations)
        )
    guard.set_read_only()

    save_state({
        "schema_version": STATE_SCHEMA_VERSION,
        "contract_id": contract.contract_id,
        "campaign_id": datetime.now(timezone.utc).strftime("c%Y%m%d%H%M%S"),
        "created_utc": utc_now(),
        "round": 0,
        "generation": 0,
        "baseline_primary": value,
        "best_primary": value,
        "best_commit": git.head(),
        "stagnation": 0,
        "tested": {},
        "last_accepted": None,
        "gate": {"incumbent_scores": {}},
        "current_generation": None,
    })
    append_jsonl(LEDGER_PATH, {
        "record_type": "baseline",
        "run_id": "baseline",
        "timestamp_utc": utc_now(),
        "commit": git.head(),
        "primary": value,
        "metrics_path": str(BASELINE_DIR / "metrics_dev.json"),
    })
    rebuild_insights()

    pm = contract.primary_metric
    print(f"[init] baseline {pm.name} (dev) = {value:.6f} at {git.head()[:12]}")
    print("[init] protected files set read-only; ready: "
          "`uv run python orchestrator.py run --generations N`")
    return 0


def build_context(args: argparse.Namespace, contract: ResearchContract,
                  git: Git, guard: ProtectionGuard) -> LoopContext:
    heuristic = HeuristicProposer()
    if args.proposer == "claude":
        proposer: Any = FallbackProposer(
            ClaudeProposer(model=args.model, max_budget_usd=args.max_budget_usd),
            heuristic,
        )
    else:
        proposer = heuristic
    coder = None
    if contract.portfolio.max_coder_hypotheses > 0 and args.proposer == "claude":
        # Coder hypotheses only arise from the LLM proposer; heuristic runs
        # stay fully offline and deterministic.
        coder = ClaudeCoder(contract, model=args.model)
    return LoopContext(contract=contract, git=git, guard=guard,
                       patcher=HyperparamsPatcher(), proposer=proposer,
                       coder=coder)


def cmd_run(args: argparse.Namespace) -> int:
    contract = load_contract()
    git = Git(ROOT)
    guard = ProtectionGuard(ROOT, contract)
    if not git.is_repo() or not git.has_head():
        raise OrchestratorError("not initialized — run `orchestrator.py init` first")

    ctx = build_context(args, contract, git, guard)
    recover(ctx)

    dirty = git.status_paths(include_untracked=False)
    if dirty:
        raise OrchestratorError(
            f"main working tree has uncommitted tracked changes {dirty}; "
            "commit or restore them first — generations branch from HEAD, so "
            "a dirty tree desynchronizes proposals from what actually runs"
        )

    pm = contract.primary_metric
    pf = contract.portfolio
    executed = 0
    stop_reason = None
    while executed < args.generations:
        state = load_state()
        if state["round"] >= contract.budgets.max_rounds:
            stop_reason = "max_rounds"
            break
        if (pf.max_generations is not None
                and state.get("generation", 0) >= pf.max_generations):
            stop_reason = "max_generations"
            break
        if state["stagnation"] >= contract.stop_conditions.stagnation_generations:
            stop_reason = "stagnation"
            break
        result = run_generation(ctx, state)
        if result is None:
            stop_reason = "search_space_exhausted"
            break
        if "stop" in result:
            stop_reason = result["stop"]
            break
        executed += 1
        gate = result["gate"]
        print(f"— generation g{gate['generation']:04d} —")
        for record in result["records"]:
            iv = (record.get("hypothesis") or {}).get("intervention") or {}
            primary = record.get("primary")
            primary_txt = f"{primary:.4f}" if isinstance(primary, float) else "n/a"
            if record.get("executor") == "coder":
                what = "coder"
            else:
                what = f"{iv.get('param')}: {iv.get('from')!r} -> {iv.get('to')!r}"
            failure = record.get("failure_class")
            print(f"  [{record['run_id']}] {what}  {pm.name}={primary_txt}  "
                  f"verdict={record['verdict']}"
                  f"{' (' + str(failure) + ')' if failure else ''}  "
                  f"decision={record['decision'].upper()}")
        # Gate outcome for humans: PASS/FAIL bits only — gate scores never
        # reach the console (transcripts get pasted into LLM contexts).
        if gate["candidates"]:
            outcome = (f"winner {result['winner']}" if result["winner"]
                       else "no candidate admitted")
            print(f"  [gate] candidates {gate['candidates']} -> {outcome}")
        else:
            print("  [gate] no dev improvers; gate skipped")

    state = load_state()
    best = state.get("best_primary")
    baseline = state.get("baseline_primary")
    print(f"\ngenerations executed: {executed} (total {state.get('generation', 0)}; "
          f"experiments {state['round']}); "
          f"stop: {stop_reason or 'requested generations done'}")
    if isinstance(best, float) and isinstance(baseline, float):
        print(f"{pm.name} (dev): baseline {baseline:.6f} -> best {best:.6f} "
              f"({(baseline - best) / baseline:+.2%} relative)"
              if pm.direction == "minimize" else
              f"{pm.name} (dev): baseline {baseline:.6f} -> best {best:.6f}")
    print(f"incumbent commit: {state.get('best_commit', '?')[:12]}  "
          f"stagnation: {state.get('stagnation')} generations")
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    contract = load_contract()
    state = load_state()
    pm = contract.primary_metric
    print(f"contract:   {contract.contract_id}")
    print(f"objective:  {contract.objective[:100]}")
    print(f"metric:     {pm.name} ({pm.direction}, "
          f"min rel improvement {pm.min_relative_improvement:.1%}; "
          f"gate epsilon {contract.portfolio.gate_min_relative_improvement:.2%})")
    print(f"experiments: {state['round']} / {contract.budgets.max_rounds}  "
          f"generations: {state.get('generation', 0)}")
    baseline = state.get("baseline_primary")
    best = state.get("best_primary")
    if isinstance(baseline, float) and isinstance(best, float):
        print(f"baseline:   {baseline:.6f} (dev)")
        print(f"best:       {best:.6f} (dev) at {state.get('best_commit', '?')[:12]}")
    print(f"stagnation: {state.get('stagnation')} / "
          f"{contract.stop_conditions.stagnation_generations} generations")
    records = read_jsonl(LEDGER_PATH)
    experiments = [r for r in records if r.get("record_type") == "experiment"]
    if experiments:
        print(f"\nlast {min(12, len(experiments))} experiments:")
        for r in experiments[-12:]:
            hyp = r.get("hypothesis") or {}
            iv = hyp.get("intervention") or {}
            primary = r.get("primary")
            primary_txt = f"{primary:.4f}" if isinstance(primary, float) else "n/a"
            gen = r.get("generation")
            gen_txt = f"g{gen:04d} " if isinstance(gen, int) else ""
            if r.get("executor") == "coder":
                what = f"coder ({(hyp.get('statement') or '')[:40]})"
            elif iv.get("param") is None:
                print(f"  {gen_txt}{r.get('run_id')}: (interrupted before "
                      f"patch)  {r.get('verdict')}  {r.get('decision')}")
                continue
            else:
                what = f"{iv.get('param')} {iv.get('from')!r}->{iv.get('to')!r}"
            print(f"  {gen_txt}{r.get('run_id')}: {what}  {primary_txt}  "
                  f"{r.get('verdict')}  {r.get('decision')}")
    gates = [r for r in records if r.get("record_type") == "gate"]
    if gates:
        print(f"\nlast {min(5, len(gates))} gate decisions (scores withheld):")
        for g in gates[-5:]:
            print(f"  g{g.get('generation', 0):04d}: candidates "
                  f"{g.get('candidates')} -> "
                  f"{g.get('winner') or 'no winner'}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """One-shot final report on the untouched test split.

    The test split is single-use by contract; a second run requires --force
    and is disclosed in test_evaluations. Nothing here feeds back into
    state, insights, or proposals.
    """
    contract = load_contract()
    git = Git(ROOT)
    guard = ProtectionGuard(ROOT, contract)
    state = load_state()
    pm = contract.primary_metric

    records = read_jsonl(LEDGER_PATH)
    prior_reports = [r for r in records if r.get("record_type") == "final_report"]
    if prior_reports and not args.force:
        raise OrchestratorError(
            f"{len(prior_reports)} final report(s) already exist — the test "
            f"split is single-use. Re-run with --force only if you accept "
            f"the multiple-testing disclosure."
        )
    baseline_rec = next((r for r in records
                         if r.get("record_type") == "baseline"), None)
    if baseline_rec is None:
        raise OrchestratorError("no baseline record in the ledger")

    report_dir = EXPERIMENTS_DIR / "report"

    def test_eval(name: str, commit: str) -> float | None:
        if git.head() == commit and not git.status_paths(include_untracked=False):
            workspace, cleanup = ROOT, False
        else:
            workspace = WORKTREES_DIR / f"report-{name}"
            git.worktree_remove(workspace)
            git.worktree_add_detached(workspace, commit)
            cleanup = True
        try:
            metrics = run_evaluator(guard, workspace, "dev",
                                    report_dir / f"{name}_test.json",
                                    f"report-{name}",
                                    contract.budgets.dev_train_timeout_s,
                                    split="test")
        finally:
            if cleanup:
                git.worktree_remove(workspace)
        return (metrics.get("primary_metric") or {}).get("value")

    baseline_test = test_eval("baseline", baseline_rec["commit"])
    incumbent_test = test_eval("incumbent", state["best_commit"])

    experiments = [r for r in records if r.get("record_type") == "experiment"]
    gates = [r for r in records if r.get("record_type") == "gate"]
    gate_eval_count = (
        sum(len(g.get("results") or {}) for g in gates)
        + sum(1 for g in gates if g.get("incumbent_evaluated"))
    )

    relative = None
    if isinstance(baseline_test, float) and isinstance(incumbent_test, float):
        relative = ((baseline_test - incumbent_test) / abs(baseline_test)
                    if pm.direction == "minimize"
                    else (incumbent_test - baseline_test) / abs(baseline_test))

    report = {
        "record_type": "final_report",
        "timestamp_utc": utc_now(),
        "campaign_id": state.get("campaign_id"),
        "contract_id": contract.contract_id,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "evaluator_sha256": sha256_file(EVALUATOR_PATH),
        "baseline_commit": baseline_rec["commit"],
        "incumbent_commit": state.get("best_commit"),
        "test": {"baseline": baseline_test, "incumbent": incumbent_test,
                 "relative_improvement": relative},
        "dev": {"baseline": state.get("baseline_primary"),
                "incumbent": state.get("best_primary")},
        "multiple_testing_disclosure": {
            "experiments": len(experiments),
            "gate_evaluations": gate_eval_count,
            "test_evaluations": len(prior_reports) + 1,
            "generations": state.get("generation", 0),
            "accepted": sum(1 for r in experiments
                            if r.get("decision") == "accept"),
            "note": "the incumbent's cached gate score was itself "
                    "gate-selected, so the admission bar errs conservative",
        },
        "selection_rule": gates[-1].get("selection_rule") if gates else None,
        "stop_state": {"stagnation": state.get("stagnation"),
                       "round": state.get("round")},
    }
    append_jsonl(LEDGER_PATH, report)
    atomic_write_json(report_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_verify_protection(_: argparse.Namespace) -> int:
    contract = load_contract()
    guard = ProtectionGuard(ROOT, contract)
    violations = guard.verify()
    if violations:
        for v in violations:
            print(f"VIOLATION: {v}")
        return 1
    files = guard.load_manifest()["files"]
    print(f"OK — {len(files)} protected files match the manifest")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="AutoResearch Phase 1 orchestrator")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="git init, baseline eval, protection manifest")
    p_init.add_argument("--force", action="store_true",
                        help="re-baseline: clear experiments/, regenerate "
                             "held-out seed and manifest")
    p_init.set_defaults(fn=cmd_init)

    p_run = sub.add_parser("run", help="execute portfolio generations")
    p_run.add_argument("--generations", type=int, default=3)
    p_run.add_argument("--proposer", choices=("heuristic", "claude"),
                       default="heuristic")
    p_run.add_argument("--model", default=None,
                       help="model override for the claude proposer/coder")
    p_run.add_argument("--max-budget-usd", type=float, default=0.5,
                       help="per-proposal budget cap for the claude proposer")
    p_run.set_defaults(fn=cmd_run)

    p_status = sub.add_parser("status", help="show campaign state")
    p_status.set_defaults(fn=cmd_status)

    p_report = sub.add_parser("report",
                              help="one-shot final report on the test split")
    p_report.add_argument("--force", action="store_true",
                          help="re-run despite an existing report (counted "
                               "in the multiple-testing disclosure)")
    p_report.set_defaults(fn=cmd_report)

    p_verify = sub.add_parser("verify-protection",
                              help="check protected files against the manifest")
    p_verify.set_defaults(fn=cmd_verify_protection)

    args = parser.parse_args()
    try:
        if args.command in ("init", "run", "report"):
            with InstanceLock(LOCK_PATH):
                return args.fn(args)
        return args.fn(args)
    except OrchestratorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
