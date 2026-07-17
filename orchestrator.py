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

STATE_SCHEMA_VERSION = 1

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
    stagnation_rounds: int


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
    if schema_version != 1:
        raise ContractError(f"unsupported contract schema_version {schema_version}")

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
        stagnation_rounds=_require(stop_raw, "stagnation_rounds", int, "stop_conditions")
    )
    if stop.stagnation_rounds <= 0:
        raise ContractError("stop_conditions.stagnation_rounds must be positive")

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
        self.run("branch", "-D", name)

    def branch_tip(self, name: str) -> str:
        return self.run("rev-parse", f"refs/heads/{name}").stdout.strip()

    def worktree_add(self, path: Path, branch: str, base: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.run("worktree", "add", "-b", branch, str(path), base)

    def worktree_remove(self, path: Path) -> None:
        proc = self.run("worktree", "remove", "--force", str(path), check=False)
        if proc.returncode != 0 and path.exists():
            shutil.rmtree(path, ignore_errors=True)
        self.run("worktree", "prune", check=False)

    def worktree_prune(self) -> None:
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
    prior_evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


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

    def propose(self, ctx: ProposalContext) -> Optional[Hypothesis]:
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
        if not candidates:
            return None

        last = ctx.last_accepted
        if last:
            for cand in candidates:
                if cand[0] == last.get("kind"):
                    candidates.remove(cand)
                    candidates.insert(0, cand)
                    break

        kind, param, new_value = candidates[0]
        pm = ctx.contract.primary_metric
        best = ctx.best_primary
        best_txt = f"{best:.4f}" if best is not None else "the baseline"
        return Hypothesis(
            id=f"h_r{ctx.round_index:04d}_{param}",
            round=ctx.round_index,
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
            prior_evidence=[i["insight_id"] for i in ctx.insights[-3:]
                            if "insight_id" in i],
        )


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

    def _schema(self, hp: dict) -> dict:
        return {
            "type": "object",
            "properties": {
                "statement": {"type": "string"},
                "mechanism": {"type": "string"},
                "param": {"type": "string", "enum": sorted(hp.keys())},
                # anyOf (not a union type list): the SDK's strict-mode schema
                # validator rejects `"type": [...]` unions.
                "new_value": {"anyOf": [{"type": "number"}, {"type": "boolean"}]},
                "predicted_effect": {"type": "string"},
                "falsifier": {"type": "string"},
            },
            "required": ["statement", "mechanism", "param", "new_value",
                         "predicted_effect", "falsifier"],
            "additionalProperties": False,
        }

    def _prompt(self, ctx: ProposalContext, feedback: str | None) -> str:
        pm = ctx.contract.primary_metric
        insights = ctx.insights[-10:]
        tested_lines = [
            f"  {param}: {', '.join(values)}"
            for param, values in sorted(ctx.tested.items())
        ] or ["  (none yet)"]
        parts = [
            "You are the hypothesis proposer inside a constrained autoresearch "
            "loop (Karpathy-style keep/reject).",
            f"Objective: {ctx.contract.objective}",
            f"Primary metric: {pm.name} ({pm.direction}); a candidate is kept "
            f"only if it improves the incumbent by >= "
            f"{pm.min_relative_improvement:.1%} relative. Evaluation is "
            f"deterministic (fixed seeds).",
            f"Current incumbent {pm.name}: {ctx.best_primary!r}",
            f"Current hyperparameters of src/train.py (minibatch SGD, linear "
            f"model, 8 features with heterogeneous scales): "
            f"{json.dumps(ctx.current_hyperparams)}",
            "Already-tested interventions — do NOT repeat any of these "
            "(param: values):",
            *tested_lines,
            "Distilled insights from past rounds:",
            json.dumps(insights, indent=2) if insights else "  (none yet)",
            "Propose exactly ONE atomic intervention: change ONE "
            "hyperparameter to ONE new value. Ground the mechanism in "
            "optimization reasoning, state a concrete falsifier, and prefer "
            "the highest-expected-improvement untested move.",
        ]
        if feedback:
            parts.append(f"Your previous proposal was rejected: {feedback}. "
                         f"Propose a different, valid intervention.")
        return "\n\n".join(parts)

    def _query(self, prompt: str, hp: dict) -> dict:
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
            output_format={"type": "json_schema", "schema": self._schema(hp)},
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

    def _validate(self, raw: dict, ctx: ProposalContext) -> tuple[str, Any] | str:
        """Returns (param, coerced_value) or an error string."""
        hp = ctx.current_hyperparams
        param = raw.get("param")
        if not isinstance(param, str) or param not in hp:
            return f"unknown param {param!r}"
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
        return (param, new_value)

    def propose(self, ctx: ProposalContext) -> Optional[Hypothesis]:
        feedback: str | None = None
        for _ in range(2):  # one retry with validation feedback
            raw = self._query(self._prompt(ctx, feedback), ctx.current_hyperparams)
            validated = self._validate(raw, ctx)
            if isinstance(validated, str):
                feedback = validated
                continue
            param, new_value = validated
            return Hypothesis(
                id=f"h_r{ctx.round_index:04d}_{param}",
                round=ctx.round_index,
                statement=str(raw["statement"]),
                mechanism=str(raw["mechanism"]),
                intervention={"param": param, "from": ctx.current_hyperparams[param],
                              "to": new_value, "kind": f"claude:{param}"},
                predicted_effect=str(raw["predicted_effect"]),
                falsifier=str(raw["falsifier"]),
                minimal_test="one smoke + one dev evaluation on the patched worktree",
                proposer=self.name,
                prior_evidence=[i["insight_id"] for i in ctx.insights[-3:]
                                if "insight_id" in i],
            )
        raise ProposerError(f"Claude proposer failed validation twice: {feedback}")


class FallbackProposer:
    """Primary proposer with heuristic fallback (never blocks the loop)."""

    def __init__(self, primary, fallback: HeuristicProposer) -> None:
        self.primary = primary
        self.fallback = fallback
        self.name = f"{primary.name}+fallback"

    @property
    def last_cost_usd(self) -> float | None:
        return getattr(self.primary, "last_cost_usd", None)

    def propose(self, ctx: ProposalContext) -> Optional[Hypothesis]:
        try:
            return self.primary.propose(ctx)
        except Exception as exc:  # SDK/network/validation failure
            print(f"[warn] {self.primary.name} proposer failed ({exc}); "
                  f"falling back to heuristic", file=sys.stderr)
            return self.fallback.propose(ctx)


# ---------------------------------------------------------------------------
# Evaluator invocation
# ---------------------------------------------------------------------------

def run_evaluator(guard: ProtectionGuard, workspace: Path, mode: str,
                  out_path: Path, run_id: str, timeout_s: int) -> dict:
    nonce = secrets.token_hex(16)
    cmd = [sys.executable, "-B", str(EVALUATOR_PATH),
           "--workspace", str(workspace), "--mode", mode,
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
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise OrchestratorError(
            "no experiments/state.json — run `orchestrator.py init` first"
        ) from None
    except json.JSONDecodeError as exc:
        raise OrchestratorError(f"state.json corrupt: {exc}") from None


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
    if record.get("record_type") != "experiment":
        return None
    hyp = record.get("hypothesis") or {}
    intervention = hyp.get("intervention", {})
    param = intervention.get("param")
    frm, to = intervention.get("from"), intervention.get("to")
    verdict = record.get("verdict")
    fc = record.get("failure_class")
    before = record.get("best_primary_before")
    primary = record.get("primary")

    before_txt = f"{before:.4f}" if isinstance(before, (int, float)) else "?"
    primary_txt = f"{primary:.4f}" if isinstance(primary, (int, float)) else "n/a"

    if verdict == "valid_positive":
        observation = (f"{param} {frm!r}->{to!r} improved heldout_rmse "
                       f"{before_txt}->{primary_txt}; accepted.")
        follow_up = f"continue moving {param} in this direction"
        confidence = 0.8
    elif fc in ("degenerate_weights", "no_skill"):
        observation = (f"{param} {frm!r}->{to!r} made training degenerate "
                       f"({fc}); optimization left the stable region.")
        follow_up = f"treat {to!r} as an upper bound for {param} in this regime"
        confidence = 0.9
    elif fc in ("timeout", "nonzero_exit"):
        observation = (f"{param} {frm!r}->{to!r} made the run fail "
                       f"({fc}) despite a valid patch — the intervention "
                       f"itself is infeasible under the budget.")
        follow_up = f"avoid pushing {param} further this way"
        confidence = 0.7
    elif verdict == "valid_negative":
        observation = (f"{param} {frm!r}->{to!r} regressed heldout_rmse "
                       f"{before_txt}->{primary_txt}; rejected.")
        follow_up = f"deprioritize this direction for {param}"
        confidence = 0.7
    elif verdict == "valid_inconclusive":
        observation = (f"{param} {frm!r}->{to!r} changed heldout_rmse "
                       f"{before_txt}->{primary_txt}, below the "
                       f"min_relative_improvement threshold.")
        follow_up = f"{param} is not the binding constraint near this optimum"
        confidence = 0.5
    elif verdict in ("invalid_implementation", "contract_violation", "aborted"):
        observation = (f"round {record.get('run_id')} ended as {verdict} "
                       f"({fc or 'no failure class'}) — no scientific signal.")
        follow_up = "no scientific conclusion; mechanical/protocol issue"
        confidence = 0.3
    else:
        return None

    return {
        "insight_id": f"ins_{record.get('run_id', '?')}",
        "scope": "sgd hyperparameter tuning on synthetic heteroscale regression",
        "round": record.get("round"),
        "hypothesis_id": hyp.get("id"),
        "observation": observation,
        "outcome": verdict,
        "failure_class": fc,
        "conditions": {"param": param, "from": frm, "to": to},
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
    stagnation = 0
    last_accepted = None
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
        if r.get("decision") == "accept" and r.get("run_id") not in corrected:
            stagnation = 0
            last_accepted = dict(intervention)
        else:
            stagnation += 1
    state["tested"] = tested
    state["stagnation"] = stagnation
    state["last_accepted"] = last_accepted


# ---------------------------------------------------------------------------
# Round execution
# ---------------------------------------------------------------------------

@dataclass
class LoopContext:
    contract: ResearchContract
    git: Git
    guard: ProtectionGuard
    patcher: HyperparamsPatcher
    proposer: Any


def _finish_round(ctx: LoopContext, state: dict, record: dict,
                  wt_path: Path, branch: str, delete_branch: bool) -> dict:
    """Merge guard, write-ahead ledger record, then merge/cleanup/state."""
    if record["decision"] == "accept":
        # Pre-flight the merge guard BEFORE the write-ahead record: an accept
        # that cannot land (main moved/dirty mid-round) must never enter the
        # ledger as accepted — it would poison insights and replay forever.
        head = ctx.git.head()
        dirty = ctx.git.status_paths(include_untracked=False)
        if head != record["base_commit"] or dirty:
            record["verdict"] = "ff_conflict"
            record["failure_class"] = (
                f"main moved during round "
                f"({record['base_commit'][:12]} -> {head[:12]})"
                if head != record["base_commit"]
                else f"main working tree dirty: {dirty}"
            )
            record["decision"] = "reject"
            record["best_primary_after"] = state.get("best_primary")

    append_jsonl(LEDGER_PATH, record)

    if record["decision"] == "accept":
        try:
            ctx.git.merge_ff(record["commit"])
        except GitError as exc:
            # Compensating record: recovery/replay must not treat this round
            # as an accepted improvement.
            append_jsonl(LEDGER_PATH, {
                "record_type": "correction",
                "corrects": record["run_id"],
                "timestamp_utc": utc_now(),
                "reason": f"ff-merge failed after accept was recorded: {exc}",
            })
            raise
        state["best_primary"] = record["primary"]
        state["best_commit"] = record["commit"]
        state["stagnation"] = 0
        state["last_accepted"] = dict(record["hypothesis"]["intervention"])
    else:
        state["stagnation"] = state.get("stagnation", 0) + 1

    # Register both endpoints as tested: the to-value was just measured, and
    # the from-value's config is the incumbent whose score is already known —
    # this also prevents proposing a plain revert of an accepted change.
    intervention = record["hypothesis"]["intervention"]
    tested_values = state.setdefault("tested", {}).setdefault(
        intervention["param"], []
    )
    for v in (value_repr(intervention["to"]), value_repr(intervention["from"])):
        if v not in tested_values:
            tested_values.append(v)

    ctx.git.worktree_remove(wt_path)
    if delete_branch and ctx.git.branch_exists(branch):
        ctx.git.delete_branch(branch)

    rebuild_insights()
    state["current_round"] = None
    save_state(state)
    return record


def run_round(ctx: LoopContext, state: dict) -> dict | None:
    """Execute one keep/reject round. Returns the ledger record, or None if
    the proposer's move space is exhausted."""
    contract = ctx.contract
    pm = contract.primary_metric

    violations = ctx.guard.verify()
    if violations:
        raise ProtectionViolation("; ".join(violations))

    current_hp = ctx.patcher.read(ROOT / TRAIN_REL)
    round_index = state["round"] + 1
    proposal = ctx.proposer.propose(ProposalContext(
        contract=contract,
        round_index=round_index,
        current_hyperparams=current_hp,
        best_primary=state.get("best_primary"),
        tested=state.get("tested", {}),
        last_accepted=state.get("last_accepted"),
        insights=read_insights(),
    ))
    if proposal is None:
        return None

    run_id = f"r{round_index:04d}"
    # Branch names are namespaced by campaign so that `init --force` (a new
    # campaign, round counter reset) never collides with retained provenance
    # branches from earlier campaigns.
    campaign = state.get("campaign_id", "c0")
    branch = f"hyp/{campaign}/{run_id}-{proposal.intervention['param']}"
    wt_path = WORKTREES_DIR / run_id
    round_dir = ROUNDS_DIR / run_id
    round_dir.mkdir(parents=True, exist_ok=True)
    base_commit = ctx.git.head()

    # Round numbers are never reused: persist the marker before any git work.
    state["round"] = round_index
    state["current_round"] = {
        "run_id": run_id, "branch": branch, "worktree": str(wt_path),
        "base_commit": base_commit, "phase": "started", "started_utc": utc_now(),
    }
    save_state(state)

    atomic_write_json(round_dir / "hypothesis.json", proposal.to_dict())

    record: dict = {
        "record_type": "experiment",
        "run_id": run_id,
        "round": round_index,
        "timestamp_utc": utc_now(),
        "hypothesis": proposal.to_dict(),
        "branch": branch,
        "base_commit": base_commit,
        "commit": None,
        "verdict": None,
        "failure_class": None,
        "decision": "reject",
        "primary": None,
        "best_primary_before": state.get("best_primary"),
        "metrics_path": None,
        "proposer": proposal.proposer,
        "proposal_cost_usd": getattr(ctx.proposer, "last_cost_usd", None)
        if proposal.proposer == "claude"
        else None,
    }

    if ctx.git.branch_exists(branch):
        raise GitError(f"branch {branch} already exists — recovery did not run?")
    ctx.git.worktree_add(wt_path, branch, base_commit)

    # --- patch, with bounded MECHANICAL repair only -------------------------
    param = proposal.intervention["param"]
    new_value = proposal.intervention["to"]
    train_file = wt_path / TRAIN_REL
    patch_error: PatchError | None = None
    for attempt in range(contract.budgets.repair_attempts + 1):
        try:
            ctx.patcher.apply(train_file, param, new_value, attempt=attempt)
            patch_error = None
            break
        except PatchError as exc:
            patch_error = exc
            ctx.git.run("checkout", "--", TRAIN_REL, cwd=wt_path, check=False)
    if patch_error is not None:
        record["verdict"] = "invalid_implementation"
        record["failure_class"] = f"patch_failed: {patch_error}"
        return _finish_round(ctx, state, record, wt_path, branch,
                             delete_branch=True)

    ctx.git.run("add", "-A", cwd=wt_path)
    commit_proc = ctx.git.run(
        "commit", "-m",
        f"{run_id}: {param} {current_hp[param]!r} -> {new_value!r}\n\n"
        f"Hypothesis: {proposal.statement}",
        cwd=wt_path, check=False,
    )
    if commit_proc.returncode != 0:
        output = commit_proc.stdout + commit_proc.stderr
        if "nothing to commit" in output or "nothing added to commit" in output:
            # No-op patch (e.g., proposed value already in HEAD): mechanical.
            record["verdict"] = "invalid_implementation"
            record["failure_class"] = "empty_diff"
            return _finish_round(ctx, state, record, wt_path, branch,
                                 delete_branch=True)
        raise GitError(f"worktree commit failed: {output.strip()}")
    commit = ctx.git.head(cwd=wt_path)
    record["commit"] = commit
    state["current_round"]["phase"] = "committed"
    save_state(state)

    changed = ctx.git.diff_paths(base_commit, cwd=wt_path)
    if not changed:
        record["verdict"] = "invalid_implementation"
        record["failure_class"] = "empty_diff"
        return _finish_round(ctx, state, record, wt_path, branch,
                             delete_branch=True)

    def glob_violations() -> list[str]:
        paths = set(ctx.git.diff_paths(base_commit, cwd=wt_path))
        paths.update(ctx.git.status_paths(cwd=wt_path))
        bad = [p for p in sorted(paths)
               if matches_any(p, contract.protected_globs)
               or not matches_any(p, contract.editable_globs)]
        return bad

    bad = glob_violations()
    if bad:
        record["verdict"] = "contract_violation"
        record["failure_class"] = f"pre-eval protected/editable violation: {bad}"
        return _finish_round(ctx, state, record, wt_path, branch,
                             delete_branch=False)

    # --- evaluate: smoke, then dev ------------------------------------------
    smoke = run_evaluator(ctx.guard, wt_path, "smoke",
                          round_dir / "metrics_smoke.json", run_id,
                          contract.budgets.smoke_train_timeout_s)
    record["smoke_metrics_path"] = str(round_dir / "metrics_smoke.json")

    if not smoke.get("executed") or smoke.get("degenerate"):
        verdict, failure_class, primary = classify(
            smoke, state.get("best_primary"), pm
        )
        record.update(verdict=verdict, failure_class=failure_class,
                      primary=primary,
                      metrics_path=str(round_dir / "metrics_smoke.json"))
    else:
        dev = run_evaluator(ctx.guard, wt_path, "dev",
                            round_dir / "metrics_dev.json", run_id,
                            contract.budgets.dev_train_timeout_s)
        verdict, failure_class, primary = classify(
            dev, state.get("best_primary"), pm
        )
        record.update(verdict=verdict, failure_class=failure_class,
                      primary=primary,
                      metrics_path=str(round_dir / "metrics_dev.json"))

    # --- post-evaluation tamper check ----------------------------------------
    bad = glob_violations()
    if bad:
        record["verdict"] = "contract_violation"
        record["failure_class"] = f"post-eval protected/editable violation: {bad}"
        record["decision"] = "reject"
        return _finish_round(ctx, state, record, wt_path, branch,
                             delete_branch=False)

    record["decision"] = "accept" if record["verdict"] == "valid_positive" else "reject"
    record["best_primary_after"] = (
        record["primary"] if record["decision"] == "accept"
        else state.get("best_primary")
    )
    return _finish_round(ctx, state, record, wt_path, branch,
                         delete_branch=False)


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

    current = state.get("current_round")
    if current:
        run_id = current.get("run_id")
        branch = current.get("branch")
        base = current.get("base_commit")
        print(f"[recover] round {run_id} was interrupted; marking aborted",
              file=sys.stderr)
        ledger_ids = {r.get("run_id") for r in read_jsonl(LEDGER_PATH)}
        if branch and git.branch_exists(branch):
            if git.branch_tip(branch) == base:
                git.delete_branch(branch)  # no commit: nothing to preserve
        if run_id not in ledger_ids:
            aborted_record = {
                "record_type": "experiment",
                "run_id": run_id,
                "round": state.get("round"),
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
        state["current_round"] = None

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

    if not git.is_repo():
        git.init_repo()
        print("[init] git repository created (branch: main)")

    if args.force and EXPERIMENTS_DIR.exists():
        shutil.rmtree(EXPERIMENTS_DIR)
        print("[init] --force: cleared experiments/")

    if not HELDOUT_CONFIG_PATH.exists() or args.force:
        seed = int.from_bytes(os.urandom(8), "big") % (2**31 - 1)
        atomic_write_json(HELDOUT_CONFIG_PATH, {
            "seed": seed,
            "size": 400,
            "created_utc": utc_now(),
            "note": "hidden held-out seed; untracked by git on purpose",
        })
        print("[init] generated evaluation/heldout_config.json (hidden seed)")

    guard.write_manifest()
    print(f"[init] protection manifest written "
          f"({len(guard.load_manifest()['files'])} files)")

    if not git.has_head():
        git.commit_all("AutoResearch Phase 1 scaffold\n\n"
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
        "baseline_primary": value,
        "best_primary": value,
        "best_commit": git.head(),
        "stagnation": 0,
        "tested": {},
        "last_accepted": None,
        "current_round": None,
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
    print(f"[init] baseline {pm.name} = {value:.6f} at {git.head()[:12]}")
    print("[init] protected files set read-only; ready: "
          "`uv run python orchestrator.py run --rounds N`")
    return 0


def build_proposer(args: argparse.Namespace):
    heuristic = HeuristicProposer()
    if args.proposer == "claude":
        return FallbackProposer(
            ClaudeProposer(model=args.model, max_budget_usd=args.max_budget_usd),
            heuristic,
        )
    return heuristic


def cmd_run(args: argparse.Namespace) -> int:
    contract = load_contract()
    git = Git(ROOT)
    guard = ProtectionGuard(ROOT, contract)
    if not git.is_repo() or not git.has_head():
        raise OrchestratorError("not initialized — run `orchestrator.py init` first")

    ctx = LoopContext(contract=contract, git=git, guard=guard,
                      patcher=HyperparamsPatcher(), proposer=build_proposer(args))
    recover(ctx)

    dirty = git.status_paths(include_untracked=False)
    if dirty:
        raise OrchestratorError(
            f"main working tree has uncommitted tracked changes {dirty}; "
            "commit or restore them first — rounds branch from HEAD, so a "
            "dirty tree desynchronizes proposals from what actually runs"
        )

    pm = contract.primary_metric
    executed = 0
    stop_reason = None
    while executed < args.rounds:
        state = load_state()
        if state["round"] >= contract.budgets.max_rounds:
            stop_reason = "max_rounds"
            break
        if state["stagnation"] >= contract.stop_conditions.stagnation_rounds:
            stop_reason = "stagnation"
            break
        record = run_round(ctx, state)
        if record is None:
            stop_reason = "search_space_exhausted"
            break
        executed += 1
        primary = record.get("primary")
        primary_txt = f"{primary:.4f}" if isinstance(primary, float) else "n/a"
        intervention = record["hypothesis"]["intervention"]
        print(f"[{record['run_id']}] {intervention['param']}: "
              f"{intervention['from']!r} -> {intervention['to']!r}  "
              f"{pm.name}={primary_txt}  verdict={record['verdict']}"
              f"{' (' + str(record['failure_class']) + ')' if record['failure_class'] else ''}"
              f"  decision={record['decision'].upper()}")

    state = load_state()
    best = state.get("best_primary")
    baseline = state.get("baseline_primary")
    print(f"\nrounds executed: {executed} (total {state['round']}); "
          f"stop: {stop_reason or 'requested rounds done'}")
    if isinstance(best, float) and isinstance(baseline, float):
        print(f"{pm.name}: baseline {baseline:.6f} -> best {best:.6f} "
              f"({(baseline - best) / baseline:+.2%} relative)"
              if pm.direction == "minimize" else
              f"{pm.name}: baseline {baseline:.6f} -> best {best:.6f}")
    print(f"incumbent commit: {state.get('best_commit', '?')[:12]}  "
          f"stagnation: {state.get('stagnation')}")
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    contract = load_contract()
    state = load_state()
    pm = contract.primary_metric
    print(f"contract:   {contract.contract_id}")
    print(f"objective:  {contract.objective[:100]}")
    print(f"metric:     {pm.name} ({pm.direction}, "
          f"min rel improvement {pm.min_relative_improvement:.1%})")
    print(f"round:      {state['round']} / {contract.budgets.max_rounds}")
    baseline = state.get("baseline_primary")
    best = state.get("best_primary")
    if isinstance(baseline, float) and isinstance(best, float):
        print(f"baseline:   {baseline:.6f}")
        print(f"best:       {best:.6f} at {state.get('best_commit', '?')[:12]}")
    print(f"stagnation: {state.get('stagnation')} / "
          f"{contract.stop_conditions.stagnation_rounds}")
    records = [r for r in read_jsonl(LEDGER_PATH)
               if r.get("record_type") == "experiment"]
    if records:
        print(f"\nlast {min(10, len(records))} experiments:")
        for r in records[-10:]:
            hyp = r.get("hypothesis") or {}
            iv = hyp.get("intervention") or {}
            primary = r.get("primary")
            primary_txt = f"{primary:.4f}" if isinstance(primary, float) else "n/a"
            if iv.get("param") is None:
                print(f"  {r.get('run_id')}: (interrupted before patch)  "
                      f"{r.get('verdict')}  {r.get('decision')}")
                continue
            print(f"  {r.get('run_id')}: {iv.get('param')} "
                  f"{iv.get('from')!r}->{iv.get('to')!r}  {primary_txt}  "
                  f"{r.get('verdict')}  {r.get('decision')}")
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

    p_run = sub.add_parser("run", help="execute keep/reject rounds")
    p_run.add_argument("--rounds", type=int, default=8)
    p_run.add_argument("--proposer", choices=("heuristic", "claude"),
                       default="heuristic")
    p_run.add_argument("--model", default=None,
                       help="model override for the claude proposer")
    p_run.add_argument("--max-budget-usd", type=float, default=0.5,
                       help="per-proposal budget cap for the claude proposer")
    p_run.set_defaults(fn=cmd_run)

    p_status = sub.add_parser("status", help="show campaign state")
    p_status.set_defaults(fn=cmd_status)

    p_verify = sub.add_parser("verify-protection",
                              help="check protected files against the manifest")
    p_verify.set_defaults(fn=cmd_verify_protection)

    args = parser.parse_args()
    try:
        if args.command in ("init", "run"):
            with InstanceLock(LOCK_PATH):
                return args.fn(args)
        return args.fn(args)
    except OrchestratorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
