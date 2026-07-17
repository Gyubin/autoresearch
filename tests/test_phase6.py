"""Phase 6a drills: the execution sandbox (Blueprint Layer 5) that isolates
the untrusted candidate trainer.

Run from the repo root:  uv run python tests/test_phase6.py
Self-contained checks (no pytest), same conventions as tests/test_phase2..5.
Everything here is offline and Docker-free: the container backend is exercised
through an injectable fake `runner`, never a real `docker` call. The live
container path is exercised separately by an actual `run` on a Docker host.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import orchestrator as orch  # noqa: E402
from sandbox import runner as sbx  # noqa: E402

FAILS: list[str] = []
SEED_MOUNT = "/w/evaluation/heldout_config.json"


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "ok  " if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


class RecordingRunner:
    """Fake subprocess.run: records argv, returns a chosen return code, and can
    raise TimeoutExpired on the first `docker run` to exercise teardown."""

    def __init__(self, returncode: int = 0, timeout_on_run: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.returncode = returncode
        self.timeout_on_run = timeout_on_run

    def __call__(self, argv, **kw):
        self.calls.append(list(argv))
        if self.timeout_on_run and argv[:2] == ["docker", "run"]:
            raise subprocess.TimeoutExpired(cmd=argv, timeout=kw.get("timeout"))

        class _Result:
            returncode = self.returncode
        return _Result()


def _cfg(backend="container", image="python:3.14-slim@sha256:abc123",
         memory_mb=256, cpus=0.5, pids_limit=64):
    return sbx.SandboxConfig(backend=backend, image=image, memory_mb=memory_mb,
                             cpus=cpus, pids_limit=pids_limit)


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def test_build_sandbox() -> None:
    sub = sbx.build_sandbox(_cfg(backend="subprocess", image=None))
    check("build: subprocess backend", type(sub).__name__ == "SubprocessSandbox"
          and sub.isolated is False and sub.backend == "subprocess")
    con = sbx.build_sandbox(_cfg())
    check("build: container backend", type(con).__name__ == "ContainerSandbox"
          and con.isolated is True and con.backend == "container")
    try:
        sbx.build_sandbox(_cfg(backend="gvisor"))
        check("build: unknown backend rejected", False, "no SandboxError")
    except sbx.SandboxError:
        check("build: unknown backend rejected", True)
    try:
        sbx.build_sandbox(_cfg(image=None))
        check("build: container without image rejected", False)
    except sbx.SandboxError:
        check("build: container without image rejected", True)
    check("build: SUPPORTED_BACKENDS", sbx.SUPPORTED_BACKENDS
          == ("subprocess", "container"))


# ---------------------------------------------------------------------------
# SubprocessSandbox regression — byte-for-byte the pre-Phase-6 host subprocess
# ---------------------------------------------------------------------------

def test_subprocess_regression(tmp: Path) -> None:
    sub = sbx.build_sandbox(_cfg(backend="subprocess", image=None))
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONHASHSEED": "0"}
    # ROOT is a valid workspace: it has src/train.py and evaluation/dataset.py
    # (train.py trains on the PUBLIC split only, so no heldout config needed).
    res = sub.run_train(ROOT, "dev", env, 90, tmp, "regress")
    check("subprocess: exit 0", res.exit_code == 0 and not res.timed_out,
          f"exit={res.exit_code} timed_out={res.timed_out}")
    check("subprocess: not isolated", res.isolated is False
          and res.backend == "subprocess")
    check("subprocess: artifacts under workspace",
          res.artifacts_dir == ROOT / "artifacts")
    check("subprocess: model.json produced",
          (res.artifacts_dir / "model.json").is_file())
    check("subprocess: logs in log_dir", res.stdout_path.parent == tmp)


# ---------------------------------------------------------------------------
# ContainerSandbox — docker argv hardening, masking, teardown, infra
# ---------------------------------------------------------------------------

def test_container_argv_hardening(tmp: Path) -> None:
    runner = RecordingRunner(returncode=0)
    con = sbx.ContainerSandbox(_cfg(), runner=runner)
    env = {"PATH": "/host/only", "PYTHONHASHSEED": "0", "AUTORESEARCH_SMOKE": "1"}
    res = con.run_train(ROOT, "smoke", env, 20, tmp, "r0007")
    argv = runner.calls[0]
    joined = " ".join(argv)

    check("container: docker run --rm --init", argv[:4] == ["docker", "run", "--rm", "--init"])
    check("container: no network", "--network" in argv
          and argv[argv.index("--network") + 1] == "none")
    check("container: read-only rootfs", "--read-only" in argv)
    check("container: drop all caps", "--cap-drop" in argv
          and argv[argv.index("--cap-drop") + 1] == "ALL")
    check("container: no-new-privileges", "no-new-privileges" in joined)
    check("container: non-root user", "--user" in argv)
    check("container: resource limits", "--memory" in argv and "--cpus" in argv
          and "--pids-limit" in argv and "256m" in joined and "64" in argv)
    check("container: host PATH not leaked", "/host/only" not in joined)
    check("container: deterministic env forwarded",
          "PYTHONHASHSEED=0" in argv and "AUTORESEARCH_SMOKE=1" in argv)
    check("container: image before command",
          "python:3.14-slim@sha256:abc123" in argv)
    check("container: runs train.py sandboxed",
          argv[-4:] == ["python", "-s", "-B", "src/train.py"])
    check("container: name sanitized", "--name" in argv
          and argv[argv.index("--name") + 1].startswith("ar-r0007"))
    check("container: reports isolated", res.isolated is True
          and res.backend == "container")


def test_container_masking(tmp: Path) -> None:
    runner = RecordingRunner(returncode=0)
    con = sbx.ContainerSandbox(_cfg(), runner=runner)
    env = {"PATH": "x", "PYTHONHASHSEED": "0"}
    res = con.run_train(ROOT, "dev", env, 90, tmp, "mask")
    argv = runner.calls[0]
    joined = " ".join(argv)

    # Workspace mounted read-only; only the fresh artifacts dir is writable.
    check("mask: workspace mounted read-only", f"{ROOT}:/w:ro" in joined)
    check("mask: artifacts writable + returned",
          f"{res.artifacts_dir}:/w/artifacts:rw" in joined
          and res.artifacts_dir == tmp / "artifacts_dev")
    # Held-out seeds masked with an empty file — absent even when workspace=ROOT.
    check("mask: seed file masked (empty file bind)",
          f"{SEED_MOUNT}:ro" in joined and ":/w:ro" in joined)
    mask_bind = [a for a in argv if a.endswith(f"{SEED_MOUNT}:ro")]
    check("mask: seed mask source is the empty mask file", bool(mask_bind)
          and ".sandbox_seed_mask" in mask_bind[0])
    check("mask: seed mask file is empty",
          (tmp / ".sandbox_seed_mask").is_file()
          and (tmp / ".sandbox_seed_mask").stat().st_size == 0)
    # Gate-score ledger masked with an ephemeral tmpfs.
    check("mask: experiments ledger masked (tmpfs)",
          "/w/experiments:rw" in joined and "--tmpfs" in argv)
    # The real seed path never appears as a readable source mount.
    check("mask: real heldout_config.json never mounted readable",
          "evaluation/heldout_config.json:ro" in joined
          and str(ROOT / "evaluation" / "heldout_config.json") not in joined)


def test_container_paths_absolute() -> None:
    # Docker -v sources must be absolute; relative bind sources become named
    # volumes and fail. docker_argv must resolve every bind source.
    con = sbx.ContainerSandbox(_cfg())
    argv = con.docker_argv(Path("rel/ws"), {"PYTHONHASHSEED": "0"},
                           Path("rel/art"), Path("rel/mask"), "ar-x")
    sources = [argv[i + 1].split(":")[0] for i, a in enumerate(argv) if a == "-v"]
    check("paths: all -v bind sources are absolute", bool(sources)
          and all(s.startswith("/") for s in sources),
          f"non-absolute: {[s for s in sources if not s.startswith('/')]}")


def test_container_infra_error(tmp: Path) -> None:
    # docker/OCI launch failures (125/126/127) are infra, not a candidate fault.
    for code in (125, 126, 127):
        runner = RecordingRunner(returncode=code)
        con = sbx.ContainerSandbox(_cfg(), runner=runner)
        res = con.run_train(ROOT, "dev", {"PYTHONHASHSEED": "0"}, 90, tmp, "infra")
        check(f"infra: docker exit {code} flagged infra_error",
              res.infra_error is True and res.exit_code == code)
    # A genuine candidate nonzero exit is NOT infra.
    runner = RecordingRunner(returncode=3)
    con = sbx.ContainerSandbox(_cfg(), runner=runner)
    res = con.run_train(ROOT, "dev", {"PYTHONHASHSEED": "0"}, 90, tmp, "cand")
    check("infra: candidate exit 3 is not infra_error",
          res.infra_error is False and res.exit_code == 3)


def test_container_timeout_teardown(tmp: Path) -> None:
    runner = RecordingRunner(timeout_on_run=True)
    con = sbx.ContainerSandbox(_cfg(), runner=runner)
    res = con.run_train(ROOT, "dev", {"PYTHONHASHSEED": "0"}, 5, tmp, "slow")
    check("timeout: flagged timed_out", res.timed_out is True
          and res.exit_code is None)
    forced = [c for c in runner.calls if c[:3] == ["docker", "rm", "-f"]]
    check("timeout: container force-removed (PID namespace torn down)",
          len(forced) == 1)
    check("timeout: force-remove targets this run's container",
          bool(forced) and forced[0][3].startswith("ar-slow"))


# ---------------------------------------------------------------------------
# Fail-closed preflight — never degrades silently to a weaker backend
# ---------------------------------------------------------------------------

def test_preflight_fail_closed() -> None:
    # subprocess is always ready (no-op).
    sbx.preflight(_cfg(backend="subprocess", image=None),
                  runner=RecordingRunner(returncode=99))
    check("preflight: subprocess is a no-op", True)

    # container happy path: docker info + image inspect both succeed.
    sbx.preflight(_cfg(), runner=RecordingRunner(returncode=0))
    check("preflight: container ready passes", True)

    def expect_error(name: str, runner) -> None:
        try:
            sbx.preflight(_cfg(), runner=runner)
            check(f"preflight: {name}", False, "no SandboxError")
        except sbx.SandboxError:
            check(f"preflight: {name}", True)

    # Daemon down: `docker info` returns nonzero.
    class DaemonDown:
        def __call__(self, argv, **kw):
            class R: returncode = 0 if argv[:2] == ["docker", "image"] else 1
            return R()
    expect_error("daemon down -> hard error", DaemonDown())

    # Image absent: info ok, `docker image inspect` returns nonzero.
    class ImageAbsent:
        def __call__(self, argv, **kw):
            class R: returncode = 1 if argv[:2] == ["docker", "image"] else 0
            return R()
    expect_error("pinned image absent -> hard error", ImageAbsent())

    # docker binary missing.
    def missing(argv, **kw):
        raise FileNotFoundError("docker")
    expect_error("docker CLI missing -> hard error", missing)

    # container backend with no image is rejected outright.
    try:
        sbx.preflight(_cfg(image=None), runner=RecordingRunner(0))
        check("preflight: container needs image", False)
    except sbx.SandboxError:
        check("preflight: container needs image", True)


# ---------------------------------------------------------------------------
# Contract v6 sandbox-block validation
# ---------------------------------------------------------------------------

def test_contract_sandbox_validation(tmp: Path) -> None:
    contract = orch.load_contract()
    sb = contract.sandbox
    check("contract: shipped sandbox default",
          contract.schema_version == 6 and sb.backend == "subprocess"
          and sb.image is None and sb.memory_mb == 512 and sb.cpus == 1.0
          and sb.pids_limit == 128)
    check("contract: sandbox/** on the protected path",
          "sandbox/**" in contract.protected_globs)

    text = orch.CONTRACT_PATH.read_text()

    def load_mutated(mutated: str):
        path = tmp / "mut.yaml"
        path.write_text(mutated)
        return orch.load_contract(path)

    def expect_reject(name: str, mutated: str) -> None:
        try:
            load_mutated(mutated)
            check(f"contract: {name} rejected", False, "no ContractError")
        except orch.ContractError:
            check(f"contract: {name} rejected", True)

    expect_reject("unknown backend",
                  text.replace("backend: subprocess", "backend: gvisor"))
    expect_reject("container without image",
                  text.replace("backend: subprocess", "backend: container"))
    expect_reject("memory_mb below floor",
                  text.replace("memory_mb: 512", "memory_mb: 63"))
    expect_reject("cpus not positive",
                  text.replace("cpus: 1.0", "cpus: 0"))
    expect_reject("pids_limit below 1",
                  text.replace("pids_limit: 128", "pids_limit: 0"))
    expect_reject("image not a string",
                  text.replace("image: null", "image: 123"))

    # container WITH a pinned image is accepted.
    ok = load_mutated(
        text.replace("backend: subprocess", "backend: container")
            .replace("image: null", 'image: "python:3.14-slim@sha256:abc"'))
    check("contract: container + pinned image accepted",
          ok.sandbox.backend == "container"
          and ok.sandbox.image == "python:3.14-slim@sha256:abc")


# ---------------------------------------------------------------------------
# No drift: evaluator declaration <-> sandbox module <-> orchestrator import
# ---------------------------------------------------------------------------

def test_no_backend_drift() -> None:
    decl = orch._load_evaluator_declarations()
    check("drift: evaluator declares sandbox backends",
          decl["sandbox_backends"] == sbx.SUPPORTED_BACKENDS)
    check("drift: orchestrator import matches",
          orch.SUPPORTED_SANDBOX_BACKENDS == sbx.SUPPORTED_BACKENDS)
    # The contract's shipped backend must be one the evaluator supports (the
    # exact cross-check cmd_init performs).
    contract = orch.load_contract()
    check("drift: contract backend supported by evaluator",
          contract.sandbox.backend in decl["sandbox_backends"])


# ---------------------------------------------------------------------------
# Evaluator side: sandbox provenance echo + unknown-backend rejection
# ---------------------------------------------------------------------------

def test_evaluator_provenance(tmp: Path) -> None:
    out = tmp / "ev.json"
    proc = subprocess.run(
        [sys.executable, "-B", str(orch.EVALUATOR_PATH),
         "--workspace", str(ROOT), "--mode", "dev", "--split", "dev",
         "--out", str(out), "--run-id", "prov", "--nonce", "n0",
         "--sandbox-backend", "subprocess"],
        capture_output=True, text=True, timeout=180)
    check("evaluator: exit 0 (subprocess backend)", proc.returncode == 0,
          proc.stderr[-400:])
    metrics = json.loads(out.read_text())
    check("evaluator: sandbox provenance echoed",
          metrics.get("sandbox") == {"backend": "subprocess", "image": None,
                                     "isolated": False})
    check("evaluator: executed with a primary value",
          metrics.get("executed") is True
          and metrics["primary_metric"]["value"] is not None)

    # An unknown backend is rejected by the evaluator's argparse allowlist.
    bad = subprocess.run(
        [sys.executable, "-B", str(orch.EVALUATOR_PATH),
         "--workspace", str(ROOT), "--mode", "dev", "--split", "dev",
         "--out", str(tmp / "bad.json"), "--run-id", "x", "--nonce", "n",
         "--sandbox-backend", "gvisor"],
        capture_output=True, text=True, timeout=60)
    check("evaluator: unknown --sandbox-backend rejected",
          bad.returncode != 0 and "sandbox-backend" in bad.stderr)


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_build_sandbox()
        test_subprocess_regression(tmp)
        test_container_argv_hardening(tmp)
        test_container_masking(tmp)
        test_container_paths_absolute()
        test_container_infra_error(tmp)
        test_container_timeout_teardown(tmp)
        test_preflight_fail_closed()
        test_contract_sandbox_validation(tmp)
        test_no_backend_drift()
        test_evaluator_provenance(tmp)

    print()
    if FAILS:
        print(f"{len(FAILS)} drill(s) FAILED:")
        for name in FAILS:
            print(f"  - {name}")
        return 1
    print("all Phase 6 unit drills passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
