#!/usr/bin/env python3
"""Protected mock evaluator for AutoResearch Phase 1.

PROTECTED FILE (listed in research_contract.yaml protected_globs).

Trust model / invariants:
  * Only the ROOT copy of this file is authoritative. The orchestrator always
    invokes <root>/evaluation/evaluate.py and points it at a candidate
    workspace via --workspace; copies of evaluation/ inside worktrees are
    never executed for scoring.
  * This evaluator trusts nothing outside evaluation/** (plus a hash-only
    echo of the contract):
      - budgets are hardcoded here; `orchestrator.py init` cross-checks them
        against research_contract.yaml once and fails fast on drift;
      - the held-out seed comes from evaluation/heldout_config.json next to
        THIS file, never from the workspace;
      - dataset code is loaded from THIS directory by absolute file path, so
        workspace code can never enter the evaluator's import path.
  * The train subprocess gets a from-scratch environment (PATH,
    PYTHONHASHSEED=0, optional AUTORESEARCH_SMOKE) and `-s -B` flags. The
    orchestrator's --nonce is echoed into metrics but never exported to the
    train subprocess, so candidate code cannot pre-craft a metrics file that
    passes the orchestrator's nonce check. (-I/-E are deliberately NOT used:
    they would make Python ignore PYTHONHASHSEED.)
  * The subprocess runs in its own session (start_new_session=True); on
    timeout the whole process group is SIGKILLed so grandchildren cannot
    outlive the budget.
  * Exit code convention: exit 0 iff a metrics file was written. Scientific
    failures (timeout, crash, degenerate weights, no-skill) still exit 0 with
    executed/degenerate/failure_class set. A nonzero exit means evaluator
    infrastructure itself broke and the round must be treated as unscored.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

EVALUATOR_VERSION = "1.0.0"
METRICS_SCHEMA_VERSION = "1.0"

# Hardcoded budgets and metric identity — see trust model above. The
# orchestrator cross-checks ALL of these against the contract at init and on
# every evaluation, so contract edits cannot silently invert classification.
TRAIN_TIMEOUT_S = {"smoke": 20, "dev": 90}
PRIMARY_METRIC_NAME = "heldout_rmse"
PRIMARY_METRIC_DIRECTION = "minimize"

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


def _tail(path: Path) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    return data[-TAIL_BYTES:].decode("utf-8", errors="replace")


def _is_finite_number(x: object) -> bool:
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(x)
    )


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


def _run_train(
    workspace: Path, mode: str, log_dir: Path
) -> tuple[int | None, bool, Path, Path]:
    """Run the candidate trainer. Returns (exit_code, timed_out, stdout, stderr)."""
    stdout_path = log_dir / f"train_stdout_{mode}.log"
    stderr_path = log_dir / f"train_stderr_{mode}.log"

    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONHASHSEED": "0",
    }
    if mode == "smoke":
        env["AUTORESEARCH_SMOKE"] = "1"

    cmd = [sys.executable, "-s", "-B", str(workspace / "src" / "train.py")]
    with open(stdout_path, "wb") as f_out, open(stderr_path, "wb") as f_err:
        proc = subprocess.Popen(
            cmd,
            cwd=workspace,
            env=env,
            stdout=f_out,
            stderr=f_err,
            start_new_session=True,
        )
        try:
            exit_code = proc.wait(timeout=TRAIN_TIMEOUT_S[mode])
            timed_out = False
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            proc.wait()
            exit_code = None
            timed_out = True
    return exit_code, timed_out, stdout_path, stderr_path


def _load_artifact(workspace: Path, notes: list[str]) -> tuple[dict | None, str | None]:
    """Validate and load artifacts/model.json. Returns (artifact, failure_class)."""
    artifact_path = workspace / "artifacts" / "model.json"
    if artifact_path.is_symlink():
        notes.append("artifact is a symlink; rejected")
        return None, "malformed_artifact"
    if not artifact_path.is_file():
        return None, "missing_artifact"
    if artifact_path.stat().st_size > MAX_ARTIFACT_BYTES:
        notes.append(f"artifact exceeds {MAX_ARTIFACT_BYTES} bytes; rejected")
        return None, "malformed_artifact"
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        notes.append(f"artifact unreadable: {exc}")
        return None, "malformed_artifact"
    if not isinstance(artifact, dict):
        notes.append("artifact is not a JSON object")
        return None, "malformed_artifact"
    return artifact, None


def _validate_model(
    artifact: dict, n_features: int, notes: list[str]
) -> tuple[dict | None, str | None]:
    """Extract a scoreable model from the artifact.

    Returns (model, failure_class); non-finite weights are a *degenerate*
    outcome (divergence is scientific evidence), structural problems are
    malformed_artifact.
    """
    weights = artifact.get("weights")
    bias = artifact.get("bias")
    if (
        not isinstance(weights, list)
        or len(weights) != n_features
        or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in weights)
        or not isinstance(bias, (int, float))
        or isinstance(bias, bool)
    ):
        notes.append("weights/bias missing or wrong shape")
        return None, "malformed_artifact"

    hp = artifact.get("hyperparams")
    if not isinstance(hp, dict):
        notes.append("hyperparams echo missing")
        return None, "malformed_artifact"

    scaling = bool(hp.get("feature_scaling"))
    means = artifact.get("feature_means")
    stds = artifact.get("feature_stds")
    if scaling:
        ok = (
            isinstance(means, list)
            and isinstance(stds, list)
            and len(means) == n_features
            and len(stds) == n_features
            and all(_is_finite_number(v) for v in means)
            and all(_is_finite_number(v) and v != 0.0 for v in stds)
        )
        if not ok:
            notes.append("feature_scaling=True but scaling stats invalid")
            return None, "malformed_artifact"

    if not all(math.isfinite(v) for v in weights) or not math.isfinite(bias):
        notes.append("non-finite weights/bias (training diverged)")
        return None, "degenerate_weights"

    return {
        "weights": [float(v) for v in weights],
        "bias": float(bias),
        "scaling": scaling,
        "means": means,
        "stds": stds,
    }, None


def _predict_rmse(
    model: dict, xs: list[list[float]], ys: list[float], n_features: int
) -> float | None:
    """RMSE of the model on (xs, ys); None if predictions go non-finite."""
    w = model["weights"]
    b = model["bias"]
    if model["scaling"]:
        m, s = model["means"], model["stds"]
        xs = [[(row[j] - m[j]) / s[j] for j in range(n_features)] for row in xs]
    sse = 0.0
    for x, y in zip(xs, ys):
        pred = b + sum(w[j] * x[j] for j in range(n_features))
        if not math.isfinite(pred):
            return None
        d = pred - y
        sse += d * d
    if not math.isfinite(sse):
        return None
    return math.sqrt(sse / len(ys))


def evaluate(workspace: Path, mode: str, run_id: str, nonce: str, out_path: Path) -> dict:
    started = time.perf_counter()
    out_path.parent.mkdir(parents=True, exist_ok=True)  # log files land here too
    notes: list[str] = []

    metrics: dict = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "mode": mode,
        "run_id": run_id,
        "nonce": nonce,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace),
        "workspace_commit": _workspace_commit(workspace),
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
        "hyperparams": None,
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

    train_started = time.perf_counter()
    exit_code, timed_out, stdout_path, stderr_path = _run_train(
        workspace, mode, out_path.parent
    )
    metrics["budget"]["train_seconds"] = round(time.perf_counter() - train_started, 3)
    metrics["train_exit_code"] = exit_code
    metrics["stdout_tail"] = _tail(stdout_path)
    metrics["stderr_tail"] = _tail(stderr_path)

    if timed_out:
        metrics["failure_class"] = "timeout"
        notes.append(f"train exceeded {TRAIN_TIMEOUT_S[mode]}s; process group killed")
        return done()
    if exit_code != 0:
        metrics["failure_class"] = "nonzero_exit"
        return done()

    metrics["executed"] = True

    artifact, failure = _load_artifact(workspace, notes)
    if failure or artifact is None:
        metrics["failure_class"] = failure or "malformed_artifact"
        return done()
    metrics["hyperparams"] = artifact.get("hyperparams")

    ds = _load_dataset_module()

    model, failure = _validate_model(artifact, ds.N_FEATURES, notes)
    if failure or model is None:
        metrics["failure_class"] = failure or "malformed_artifact"
        metrics["degenerate"] = failure == "degenerate_weights"
        return done()

    xs_h, ys_h = ds.load_heldout(HELDOUT_CONFIG)
    metrics["dataset"] = {
        "train_seed": ds.TRAIN_SEED,
        "n_train": ds.N_TRAIN,
        "n_heldout": len(ys_h),
        "heldout_fingerprint": ds.fingerprint(ys_h),
    }

    heldout_rmse = _predict_rmse(model, xs_h, ys_h, ds.N_FEATURES)
    if heldout_rmse is None:
        metrics["failure_class"] = "degenerate_weights"
        metrics["degenerate"] = True
        notes.append("non-finite held-out predictions")
        return done()

    mean_y = sum(ys_h) / len(ys_h)
    constant_baseline_rmse = math.sqrt(
        sum((y - mean_y) * (y - mean_y) for y in ys_h) / len(ys_h)
    )

    xs_t, ys_t = ds.load_train()
    train_rmse_recomputed = _predict_rmse(model, xs_t, ys_t, ds.N_FEATURES)

    claimed = artifact.get("train_rmse")
    metrics["metrics"] = {
        "heldout_rmse": heldout_rmse,
        "train_rmse_claimed": claimed if _is_finite_number(claimed) else None,
        "train_rmse_recomputed": train_rmse_recomputed,
        "generalization_gap": (
            heldout_rmse - train_rmse_recomputed
            if train_rmse_recomputed is not None
            else None
        ),
        "constant_baseline_rmse": constant_baseline_rmse,
    }

    if heldout_rmse > constant_baseline_rmse:
        # Finite but worse than predicting the mean: no measurable skill.
        metrics["failure_class"] = "no_skill"
        metrics["degenerate"] = True
        notes.append(
            f"heldout_rmse {heldout_rmse:.4f} worse than constant baseline "
            f"{constant_baseline_rmse:.4f}"
        )
        return done()

    metrics["primary_metric"]["value"] = heldout_rmse
    return done()


def _atomic_write(out_path: Path, payload: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--mode", choices=("smoke", "dev"), default="dev")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--nonce", default="")
    args = parser.parse_args()

    try:
        metrics = evaluate(
            args.workspace.resolve(), args.mode, args.run_id, args.nonce, args.out
        )
    except Exception as exc:  # infra failure: report loudly, exit nonzero
        crash = {
            "schema_version": METRICS_SCHEMA_VERSION,
            "mode": args.mode,
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
