#!/usr/bin/env python3
"""Execution sandbox for the untrusted candidate trainer (Phase 6a, Layer 5).

PROTECTED FILE (listed in research_contract.yaml protected_globs).

Only the candidate trainer's EXECUTION is isolated. Scoring stays in the
trusted root evaluator, which holds the held-out seeds; the seeds are never
mounted into the container (they are masked even when the workspace is the
repo ROOT), so isolated candidate code cannot read them.

Closure rules (trusted path — same discipline as literature/ and assurance/):
  * stdlib only; imports nothing from this project.
  * no import-time side effects; the module performs no file IO of its own
    beyond creating the candidate's fresh artifacts/log scratch dirs and
    launching the trainer subprocess.
  * loaded by the ROOT evaluator via absolute file path (never via sys.path),
    so a workspace copy can never shadow it.

Two backends implement the same `Sandbox` protocol:
  * SubprocessSandbox — byte-identical to the pre-Phase-6 host subprocess
    (no OS isolation). The default; keeps machines without Docker and the
    self-contained test harness working exactly as before.
  * ContainerSandbox — `docker run` with no network, a read-only rootfs,
    dropped capabilities, a non-root user, resource limits, an ephemeral
    (`--rm`) PID namespace, and the held-out seed file / experiments ledger
    masked out of the mount. Never falls back to the subprocess backend
    silently: a missing daemon or image is a hard error (see `preflight`).
"""

from __future__ import annotations

import dataclasses
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Callable, Protocol

SUPPORTED_BACKENDS = ("subprocess", "container")

# Container-internal layout. The workspace is mounted read-only here; the only
# writable surfaces are the artifacts bind mount, a /tmp tmpfs, and the masked
# experiments tmpfs.
_WORKDIR = "/w"
# Phase 6c: the trusted evaluator hands the candidate solver the split's problem
# instances (COORDINATES ONLY — never the seed) through a read-only mount at this
# fixed in-container path, advertised via AUTORESEARCH_INSTANCES. The seed masks
# below are unchanged: the solver sees the instance it must solve, never the seed
# that generated it or any other split.
_INSTANCES_MOUNT = "/w/instances.json"
# nobody:nogroup on typical Linux images — a non-root uid with no host mapping.
_UID_GID = "65534:65534"
_INFRA_EXIT_CODES = frozenset({125, 126, 127})  # docker/OCI launch failures


class SandboxError(RuntimeError):
    """Raised when the sandbox cannot be constructed or is not ready.

    Deliberately distinct from a candidate's own failure: a SandboxError means
    the isolation infrastructure is unavailable, not that the trainer ran and
    failed. Callers surface it as an evaluator-infrastructure fault, never as a
    scientific verdict — and NEVER by degrading to a weaker backend.
    """


@dataclasses.dataclass(frozen=True)
class SandboxConfig:
    backend: str
    image: str | None
    memory_mb: int
    cpus: float
    pids_limit: int
    # When True, the orchestrator refuses to score the seed-holding splits
    # (gate/test) under a non-isolating backend instead of merely warning — the
    # candidate runs on held-out instances, so only `container` (which masks the
    # seed file) yields a trust-grade score. Default False keeps the Docker-free
    # subprocess workflow working; the orchestrator warns loudly in that case.
    require_container_for_trusted_splits: bool = False


@dataclasses.dataclass(frozen=True)
class TrainResult:
    exit_code: int | None
    timed_out: bool
    stdout_path: Path
    stderr_path: Path
    artifacts_dir: Path   # where the evaluator reads artifacts/model.json from
    isolated: bool
    backend: str
    infra_error: bool = False  # container launch failed (not a candidate fault)


class Sandbox(Protocol):
    isolated: bool
    backend: str

    def run_train(self, workspace: Path, mode: str, env: dict[str, str],
                  timeout_s: int, log_dir: Path, name_hint: str,
                  instances_path: Path | None = None) -> TrainResult:
        """Run <workspace>/src/train.py, returning a TrainResult.

        `env` is the from-scratch environment the trusted evaluator built;
        `log_dir` receives the stdout/stderr logs (and, for the container
        backend, the fresh writable artifacts dir). `instances_path`, when given
        (Phase 6c), is a host file of problem instances made available to the
        solver read-only (mounted for the container backend, an env-var path for
        subprocess) via AUTORESEARCH_INSTANCES — coordinates only, no seed.
        Deterministic and side effect free apart from those scratch writes."""
        ...


class SubprocessSandbox:
    """Pre-Phase-6 behaviour: a host subprocess in its own session, SIGKILLed
    (process group) on timeout. NO OS isolation — the trainer runs with the
    invoking user's privileges. `sandbox.backend: subprocess` selects this."""

    isolated = False
    backend = "subprocess"

    def run_train(self, workspace: Path, mode: str, env: dict[str, str],
                  timeout_s: int, log_dir: Path, name_hint: str,
                  instances_path: Path | None = None) -> TrainResult:
        stdout_path = log_dir / f"train_stdout_{mode}.log"
        stderr_path = log_dir / f"train_stderr_{mode}.log"
        cmd = [sys.executable, "-s", "-B", str(workspace / "src" / "train.py")]
        if instances_path is not None:
            # Copy so we never mutate the evaluator's env dict. No OS isolation
            # here, so the solver could read other files anyway; the trusted
            # evaluator recomputes the objective, and gate/test admission is
            # only trustworthy under the container backend (documented).
            env = {**env, "AUTORESEARCH_INSTANCES": str(instances_path.resolve())}
        with open(stdout_path, "wb") as f_out, open(stderr_path, "wb") as f_err:
            proc = subprocess.Popen(
                cmd, cwd=workspace, env=env,
                stdout=f_out, stderr=f_err, start_new_session=True,
            )
            try:
                exit_code = proc.wait(timeout=timeout_s)
                timed_out = False
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                proc.wait()
                exit_code = None
                timed_out = True
        return TrainResult(
            exit_code=exit_code, timed_out=timed_out,
            stdout_path=stdout_path, stderr_path=stderr_path,
            artifacts_dir=workspace / "artifacts",
            isolated=False, backend="subprocess",
        )


def _sanitize_name(hint: str) -> str:
    """A docker --name must match [a-zA-Z0-9][a-zA-Z0-9_.-]*."""
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]", "-", hint)
    cleaned = cleaned.lstrip("-._") or "run"
    return f"ar-{cleaned}"[:120]


class ContainerSandbox:
    """`docker run`-based OS isolation for the untrusted trainer.

    The workspace is mounted read-only; the held-out seed file and the
    experiments ledger are masked so they are absent even when the workspace is
    the repo ROOT. Network is off, capabilities are dropped, the rootfs is
    read-only, and the container is ephemeral so no background process can
    outlive the evaluation (TOCTOU). On Linux a `--runtime=runsc` (gVisor) drop
    -in only adds a flag; the mount/security model is unchanged.
    """

    isolated = True
    backend = "container"

    def __init__(self, cfg: SandboxConfig,
                 runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
                 workdir: str = _WORKDIR, uid_gid: str = _UID_GID) -> None:
        if cfg.backend != "container":
            raise SandboxError(f"ContainerSandbox got backend {cfg.backend!r}")
        if not cfg.image:
            raise SandboxError("sandbox.backend 'container' requires a pinned image")
        self._cfg = cfg
        self._runner = runner
        self._workdir = workdir
        self._uid_gid = uid_gid

    def docker_argv(self, workspace: Path, env: dict[str, str],
                    host_artifacts: Path, mask_file: Path,
                    container_name: str,
                    instances_path: Path | None = None) -> list[str]:
        c = self._cfg
        wd = self._workdir
        # Docker `-v` sources must be ABSOLUTE host paths — a relative path is
        # silently read as a named-volume reference and fails. Resolve every
        # bind source so a relative workspace/out path can never slip through.
        workspace = workspace.resolve()
        host_artifacts = host_artifacts.resolve()
        mask_file = mask_file.resolve()
        argv = [
            "docker", "run", "--rm", "--init", "--name", container_name,
            "--network", "none",                       # no exfiltration/phone-home
            "--read-only",                             # immutable rootfs
            "--tmpfs", "/tmp:rw,size=64m,noexec,nosuid,nodev",
            "--user", self._uid_gid,                   # non-root
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--memory", f"{c.memory_mb}m",
            "--memory-swap", f"{c.memory_mb}m",        # deny swap escape
            "--cpus", str(c.cpus),
            "--pids-limit", str(c.pids_limit),
            # Workspace read-only; public dataset code rides along, seeds do not.
            "-v", f"{workspace}:{wd}:ro",
            # Only writable workspace surface: the fresh artifacts dir.
            "-v", f"{host_artifacts}:{wd}/artifacts:rw",
            # Mask the held-out seeds with an empty file — absent even when the
            # workspace is ROOT (which physically contains heldout_config.json).
            "-v", f"{mask_file}:{wd}/evaluation/heldout_config.json:ro",
            # Mask the experiments ledger (gate scores) with an ephemeral tmpfs.
            "--tmpfs", f"{wd}/experiments:rw,size=1m,noexec,nosuid,nodev",
            "--workdir", wd,
        ]
        # Phase 6c: the split's instances, read-only, at a fixed path. Coordinates
        # only — the seed that generated them stays in the trusted evaluator and
        # is still masked above, so the solver cannot regenerate other splits.
        if instances_path is not None:
            argv += ["-v", f"{instances_path.resolve()}:{_INSTANCES_MOUNT}:ro"]
        # Only forward the deterministic knobs; the image supplies PATH etc.
        for key in ("PYTHONHASHSEED", "AUTORESEARCH_SMOKE"):
            if key in env:
                argv += ["-e", f"{key}={env[key]}"]
        if instances_path is not None:
            argv += ["-e", f"AUTORESEARCH_INSTANCES={_INSTANCES_MOUNT}"]
        argv += [c.image, "python", "-s", "-B", "src/train.py"]
        return argv

    def run_train(self, workspace: Path, mode: str, env: dict[str, str],
                  timeout_s: int, log_dir: Path, name_hint: str,
                  instances_path: Path | None = None) -> TrainResult:
        stdout_path = log_dir / f"train_stdout_{mode}.log"
        stderr_path = log_dir / f"train_stderr_{mode}.log"
        host_artifacts = log_dir / f"artifacts_{mode}"
        if host_artifacts.exists():
            shutil.rmtree(host_artifacts)
        host_artifacts.mkdir(parents=True)
        mask_file = log_dir / ".sandbox_seed_mask"
        mask_file.write_bytes(b"")  # empty: masks the seeds if read
        container_name = _sanitize_name(f"{name_hint}-{mode}")
        argv = self.docker_argv(workspace, env, host_artifacts, mask_file,
                                container_name, instances_path=instances_path)

        exit_code: int | None = None
        timed_out = False
        infra_error = False
        with open(stdout_path, "wb") as f_out, open(stderr_path, "wb") as f_err:
            try:
                proc = self._runner(argv, stdout=f_out, stderr=f_err,
                                    timeout=timeout_s)
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                # `--rm` only removes a container that exited on its own; force
                # remove so the PID namespace (and any spawned process) dies.
                self._force_remove(container_name)
        if not timed_out and exit_code in _INFRA_EXIT_CODES:
            # docker/OCI could not launch the container — infra fault, not a
            # candidate verdict. (preflight already checked daemon+image, so
            # this is rare, e.g. a transient daemon error.)
            infra_error = True
        return TrainResult(
            exit_code=exit_code, timed_out=timed_out,
            stdout_path=stdout_path, stderr_path=stderr_path,
            artifacts_dir=host_artifacts,
            isolated=True, backend="container", infra_error=infra_error,
        )

    def _force_remove(self, container_name: str) -> None:
        try:
            self._runner(["docker", "rm", "-f", container_name],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        timeout=30)
        except (subprocess.SubprocessError, OSError):
            pass


def build_sandbox(
    cfg: SandboxConfig,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> Sandbox:
    if cfg.backend == "subprocess":
        return SubprocessSandbox()
    if cfg.backend == "container":
        return ContainerSandbox(cfg, runner=runner)
    raise SandboxError(
        f"unsupported sandbox backend {cfg.backend!r} "
        f"(supported: {', '.join(SUPPORTED_BACKENDS)})")


def preflight(
    cfg: SandboxConfig,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> None:
    """Fail-closed readiness check. No-op for subprocess.

    For container: verify the daemon is reachable and the pinned image is
    already present locally (runs are `--network none`, so an absent image
    cannot be pulled on demand). Raises SandboxError with an actionable message
    — the caller must NOT degrade to a weaker backend, or the isolation
    guarantee is silently void.
    """
    if cfg.backend == "subprocess":
        return
    if cfg.backend != "container":
        raise SandboxError(
            f"unsupported sandbox backend {cfg.backend!r} "
            f"(supported: {', '.join(SUPPORTED_BACKENDS)})")
    if not cfg.image:
        raise SandboxError("sandbox.backend 'container' requires a pinned image")
    try:
        info = runner(["docker", "info"], stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, timeout=30)
    except FileNotFoundError:
        raise SandboxError(
            "sandbox.backend 'container' needs the `docker` CLI on PATH") from None
    except subprocess.TimeoutExpired:
        raise SandboxError(
            "`docker info` timed out — is the daemon up? "
            "(colima start / Docker Desktop)") from None
    if info.returncode != 0:
        raise SandboxError(
            "Docker daemon not reachable — start it (colima start / Docker "
            "Desktop), or set sandbox.backend: subprocess in the contract")
    try:
        img = runner(["docker", "image", "inspect", cfg.image],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=30)
    except (subprocess.SubprocessError, OSError) as exc:
        raise SandboxError(f"could not inspect sandbox image: {exc}") from None
    if img.returncode != 0:
        raise SandboxError(
            f"pinned sandbox image absent — run `docker pull {cfg.image}` first "
            f"(runs are --network none, so the image cannot be pulled on demand)")
